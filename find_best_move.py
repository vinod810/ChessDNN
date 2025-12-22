import chess
import math
import time
from collections import defaultdict
import chess.polyglot

from predict_score import dnn_evaluation

# ---------------- Global Engine Configuration ----------------
CONFIG = {
    # ---------------- Basic Search ----------------
    "SEARCH_DEPTH": 4,          # Default search depth per alpha-beta call
    "ID_MAX_DEPTH": 6,          # Maximum depth for iterative deepening
    "MAX_QDEPTH": 3,            # Max quiescence search depth
    "QUIESCENCE_EXTENSION_DEPTH": 1,  # Quiescence tactical extensions

    # ---------------- LMR (Late Move Reductions) ----------------
    "LMR_ENABLED": True,
    "LMR_MIN_DEPTH": 3,         # LMR applies only at depth >= 3
    "LMR_MOVE_THRESHOLD": 3,    # Only moves beyond first 3 considered for reduction
    "LMR_REDUCTION": 1,         # Reduce depth by 1 ply

    # ---------------- Null-Move Pruning ----------------
    "NULL_MOVE_ENABLED": True,
    "NULL_MOVE_REDUCTION": 2,   # Reduce depth by 2 for null-move
    "NULL_MOVE_MIN_DEPTH": 3,   # Only apply at depth >= 3

    # ---------------- Move Ordering ----------------
    "MAX_PLY": 256,
    "EXACT": 0,
    "LOWERBOUND": 1,
    "UPPERBOUND": 2,
    "KILLER_SLOTS": 2,          # Number of killer moves stored per ply

    # ---------------- Evaluation ----------------
    "USE_LIGHTWEIGHT_QUIESCENCE": True,  # Use hybrid material+heuristic eval in quiescence
    "PIECE_VALUES": {             # Material scores
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    },

    # ---------------- Time Control ----------------
    "TIME_CONTROL_ENABLED": True,
    "TIME_LIMIT": 30.0,          # Seconds per move
    "TIME_BUFFER": 0.05,        # Safety buffer to stop search slightly early

    # ---------------- Transposition Table ----------------
    "TT_MAX_SIZE": 1_000_000,   # Maximum entries, adjust to available memory

    # ---------------- KPI Tracking ----------------
    "TRACK_KPI": True
}

# ---------------- Global Tables ----------------
transposition_table = {}
eval_cache = {}
killer_moves = [[None] * CONFIG["KILLER_SLOTS"] for _ in range(CONFIG["MAX_PLY"])]
history_table = defaultdict(int)

# ---------------- KPI Tracking ----------------
KPI = {
    "nodes_visited": 0,
    "quiescence_nodes": 0,
    "beta_cutoffs": 0,
    "alpha_cutoffs": 0,
    "tt_hits": 0,
    "dnn_evaluations": 0,
    "pv_nodes": 0,
    "null_move_prunes": 0,
    "quiescence_extensions": 0,
    "per_depth": defaultdict(lambda: {"nodes_visited": 0, "quiescence_nodes": 0, "dnn_evaluations": 0})
}

# ---------------- Evaluation Functions ----------------
def material_evaluate(board: chess.Board) -> int:
    score = 0
    for sq, piece in board.piece_map().items():
        val = CONFIG["PIECE_VALUES"][piece.piece_type]
        score += val if piece.color == chess.WHITE else -val
    return score

# def lightweight_hybrid_evaluate(board: chess.Board) -> int:
#     score = material_evaluate(board)
#     central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
#     for sq in central_squares:
#         piece = board.piece_at(sq)
#         if piece:
#             score += 10 if piece.color == chess.WHITE else -10
#     return score

def evaluate_with_cache_white_centric(board: chess.Board, depth: int = None) -> float:
    key = chess.polyglot.zobrist_hash(board) #board.zobrist_hash()
    if key in eval_cache:
        return eval_cache[key]
    score = dnn_evaluation(board)  # User-provided DNN function
    eval_cache[key] = score
    KPI["dnn_evaluations"] += 1
    if depth is not None:
        KPI["per_depth"][depth]["dnn_evaluations"] += 1
    return score

# ---------------- Move Ordering ----------------
def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return -1
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if victim is None or attacker is None:
        return -1
    return CONFIG["PIECE_VALUES"][victim.piece_type] * 100 - CONFIG["PIECE_VALUES"][attacker.piece_type]

def order_moves(board: chess.Board, pv_move: chess.Move = None, tt_move: chess.Move = None, ply: int = 0):
    legal = list(board.legal_moves)
    if pv_move in legal:
        yield pv_move
    if tt_move in legal and tt_move != pv_move:
        yield tt_move
    for km in killer_moves[ply]:
        if km in legal and km != pv_move and km != tt_move and not board.is_capture(km):
            yield km

    captures, quiets = [], []
    for mv in legal:
        if mv == pv_move or mv == tt_move or mv in killer_moves[ply]:
            continue
        if board.is_capture(mv) or mv.promotion:
            captures.append(mv)
        else:
            quiets.append(mv)

    captures.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)
    quiets.sort(key=lambda m: history_table[(1 if board.turn == chess.WHITE else 0, m.from_square, m.to_square)],
                reverse=True)

    for mv in captures + quiets:
        yield mv

# ---------------- Transposition Table ----------------
def probe_tt(board: chess.Board):
    return transposition_table.get(chess.polyglot.zobrist_hash(board), None)

def store_tt(board: chess.Board, depth: int, score: float, flag: int, move: chess.Move):
    transposition_table[chess.polyglot.zobrist_hash(board)] = {"depth": depth, "score": score, "flag": flag, "move": move}

# ---------------- Quiescence Search ----------------
def quiescence(board: chess.Board, alpha: float, beta: float, qdepth: int, ply: int, ext_depth: int = 0):
    KPI["quiescence_nodes"] += 1
    KPI["per_depth"][qdepth]["quiescence_nodes"] += 1

    in_check = board.is_check()
    best_score = -math.inf
    best_pv = []

    if not in_check:
        stand_pat_white = material_evaluate(board)
        stand_pat = stand_pat_white if board.turn == chess.WHITE else -stand_pat_white
        if stand_pat >= beta:
            KPI["beta_cutoffs"] += 1
            return stand_pat, []
        if stand_pat > alpha:
            alpha = stand_pat

    if qdepth >= CONFIG["MAX_QDEPTH"] and ext_depth >= CONFIG["QUIESCENCE_EXTENSION_DEPTH"] and not in_check:
        return alpha, []

    for move in order_moves(board, pv_move=None, tt_move=None, ply=ply):
        is_tactical = board.is_capture(move) or move.promotion or board.gives_check(move)
        if not in_check and not is_tactical:
            continue

        next_ext_depth = ext_depth
        if move.promotion or board.gives_check(move):
            next_ext_depth += 1
            KPI["quiescence_extensions"] += 1

        san = board.san(move)
        next_qdepth = qdepth + 1 if not (move.promotion or board.gives_check(move)) else qdepth # avoid branch explosion
        board.push(move)
        score, child_pv = quiescence(board, -beta, -alpha, next_qdepth, ply + 1, next_ext_depth)
        score = -score
        board.pop()

        if score >= beta:
            KPI["beta_cutoffs"] += 1
            return score, [san]
        if score > best_score:
            best_score = score
            best_pv = [san] + child_pv
        if score > alpha:
            alpha = score

    return alpha, best_pv

# ---------------- Alpha-Beta with LMR + Null-Move ----------------
def alpha_beta(board: chess.Board, depth: int = None, alpha: float = -math.inf, beta: float = math.inf, ply: int = 0,
               is_pv_node: bool = True):
    if depth is None:
        depth = CONFIG["SEARCH_DEPTH"]

    KPI["nodes_visited"] += 1
    KPI["per_depth"][depth]["nodes_visited"] += 1
    if is_pv_node:
        KPI["pv_nodes"] += 1

    # Null Move Pruning
    if CONFIG["NULL_MOVE_ENABLED"] and depth >= CONFIG["NULL_MOVE_MIN_DEPTH"] and not board.is_check():
        board.turn = not board.turn
        null_depth = depth - CONFIG["NULL_MOVE_REDUCTION"] - 1
        score, _ = alpha_beta(board, null_depth, -beta, -beta + 1, ply + 1, is_pv_node=False)
        score = -score
        board.turn = not board.turn
        if score >= beta:
            KPI["null_move_prunes"] += 1
            KPI["beta_cutoffs"] += 1
            return beta, []

    # TT Lookup
    tt_entry = probe_tt(board)
    if tt_entry and tt_entry["depth"] >= depth:
        KPI["tt_hits"] += 1
        flag = tt_entry["flag"]
        score = tt_entry["score"]
        if flag == CONFIG["EXACT"]:
            pv_moves = [board.san(tt_entry["move"])] if tt_entry["move"] else []
            return score, pv_moves
        if flag == CONFIG["LOWERBOUND"]:
            alpha = max(alpha, score)
        elif flag == CONFIG["UPPERBOUND"]:
            beta = min(beta, score)
        if alpha >= beta:
            KPI["beta_cutoffs"] += 1
            pv_moves = [board.san(tt_entry["move"])] if tt_entry["move"] else []
            return score, pv_moves

    if board.is_game_over():
        if board.is_checkmate():
            return -9999999, []
        else:
            return 0, []

    if depth == 0:
        tactical = board.is_check() or any(board.is_capture(mv) or mv.promotion or board.gives_check(mv)
                                           for mv in board.legal_moves)
        if tactical:
            #return quiescence(board, alpha, beta, qdepth=0, ply=ply)
            stand_pat = material_evaluate(board)
            q_score, q_pv = quiescence(board, alpha, beta, 0, ply)
            q_is_tactical = (q_pv != []) or (q_score != stand_pat)
            if not q_is_tactical:
                dnn_score = evaluate_with_cache_white_centric(board)
                if board.turn == chess.BLACK: dnn_score = -dnn_score
                #alpha = max(alpha, dnn_score)
                #if dnn_score >= beta: return beta, []
                #return alpha, []
                return dnn_score, []
            return q_score, q_pv
        else:
            white_score = evaluate_with_cache_white_centric(board, depth=depth)
            return (white_score if board.turn == chess.WHITE else -white_score), []

    original_alpha = alpha
    best_score = -math.inf
    best_move = None
    best_pv = []

    pv_move = tt_entry["move"] if tt_entry else None
    tt_move = pv_move

    move_number = 0
    for move in order_moves(board, pv_move=pv_move, tt_move=tt_move, ply=ply):
        move_number += 1
        san = board.san(move)
        board.push(move)
        is_tactical = board.is_capture(move) or move.promotion or board.gives_check(move)

        # LMR
        if CONFIG["LMR_ENABLED"] and depth >= CONFIG["LMR_MIN_DEPTH"] and move_number > CONFIG["LMR_MOVE_THRESHOLD"] \
           and not is_tactical and move != pv_move and move != tt_move:
            lmr_depth = max(1, depth - CONFIG["LMR_REDUCTION"])
            score, child_pv = alpha_beta(board, lmr_depth, -beta, -alpha, ply + 1, is_pv_node=False)
            score = -score
            if score > alpha:
                score, child_pv = alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, is_pv_node=True)
                score = -score
        else:
            score, child_pv = alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, is_pv_node=True)
            score = -score

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
            best_pv = [san] + child_pv
        if score > alpha:
            alpha = score

        if alpha >= beta:
            KPI["beta_cutoffs"] += 1
            if not board.is_capture(move) and move.promotion is None:
                if CONFIG["KILLER_SLOTS"] > 0 and killer_moves[ply][0] != move:
                    killer_moves[ply][1] = killer_moves[ply][0]
                    killer_moves[ply][0] = move
                side_int = 1 if board.turn == chess.WHITE else 0
                history_table[(side_int, move.from_square, move.to_square)] += depth * depth
            store_tt(board, depth, best_score, CONFIG["LOWERBOUND"], best_move)

            if best_move is None:
                # No legal moves → checkmate or stalemate
                return best_score, []
            return best_score, [board.san(best_move)]

    flag = CONFIG["EXACT"]
    if best_score <= original_alpha:
        flag = CONFIG["UPPERBOUND"]
    elif best_score >= beta:
        flag = CONFIG["LOWERBOUND"]

    store_tt(board, depth, best_score, flag, best_move)
    return best_score, best_pv

# ---------------- Find Best Move ----------------
def find_best_move(board: chess.Board):
    score, pv = alpha_beta(board, depth=CONFIG["SEARCH_DEPTH"])
    best_move = None
    if pv:
        try:
            bcopy = board.copy()
            best_move = bcopy.parse_san(pv[0])
        except:
            best_move = None
    return best_move, pv, score

# ---------------- Iterative Deepening with Time Control ----------------
def iterative_deepening_with_time(board: chess.Board):
    best_move = None
    best_pv = []
    best_score = None

    start_time = time.time()

    for depth in range(1, CONFIG["ID_MAX_DEPTH"] + 1):
        CONFIG["SEARCH_DEPTH"] = depth

        elapsed = time.time() - start_time
        if CONFIG["TIME_CONTROL_ENABLED"] and elapsed + CONFIG["TIME_BUFFER"] >= CONFIG["TIME_LIMIT"]:
            print(f"Time limit reached at depth {depth-1}")
            break

        move, pv, score = find_best_move(board)
        if move is not None:
            best_move = move
            best_pv = pv
            best_score = score

        print(f"ID Depth {depth}: Best move {best_move}, Score {best_score}, PV {best_pv}, Time elapsed {elapsed:.2f}s")

    return best_move, best_pv, best_score

# # ---------------- Example Usage ----------------
# if __name__ == "__main__":
#     board = chess.Board()
#     KPI["per_depth"] = defaultdict(lambda: {"nodes_visited": 0, "quiescence_nodes": 0, "dnn_evaluations": 0})
#
#     best_move, pv, score = iterative_deepening_with_time(board)
#
#     print("\nFinal Best move:", best_move)
#     print("Final PV:", pv)
#     print("Final Score:", score)
#     print("\nGlobal KPIs:")
#     for k, v in KPI.items():
#         if k != "per_depth":
#             print(f"{k}: {v}")
#     print("\nPer-depth KPIs:")
#     for depth in sorted(KPI["per_depth"].keys(), reverse=True):
#         stats = KPI["per_depth"][depth]
#         print(f"Depth {depth}: Nodes={stats['nodes_visited']}, Quiescence={stats['quiescence_nodes']}, "
#               f"DNN calls={stats['dnn_evaluations']}")


def main():

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break

            board = chess.Board(fen)
            KPI["per_depth"] = defaultdict(lambda: {"nodes_visited": 0, "quiescence_nodes": 0, "dnn_evaluations": 0})

            best_move, pv, score = iterative_deepening_with_time(board)

            print("\nFinal Best move:", best_move)
            print("Final PV:", pv)
            print("Final Score:", score)
            print("\nGlobal KPIs:")
            for k, v in KPI.items():
                if k != "per_depth":
                    print(f"{k}: {v}")
            print("\nPer-depth KPIs:")
            for depth in sorted(KPI["per_depth"].keys(), reverse=True):
                stats = KPI["per_depth"][depth]
                print(
                    f"Depth {depth}: Nodes={stats['nodes_visited']}, Quiescence={stats['quiescence_nodes']}, DNN calls={stats['dnn_evaluations']}")

        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break

if __name__ == '__main__':
    main()
