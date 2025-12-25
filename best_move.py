import time
import chess
import chess.polyglot
from collections import namedtuple

from material import get_material_eval
from predict_score import dnn_evaluation
from prepare_data import is_capture

# ----------------------------------
# Configuration
# ----------------------------------

MAX_DEPTH = 4
INF = 10_000
MAX_TT_SIZE = 200_000
MAX_ET_SIZE = 200_000
DNN_MAX_Q_DEPTH = 20
DNN_SCORE_DIFF_THRESH = 50
TACTICAL_Q_DEPTH = 5
MAX_DNN_EAVALS = 300
ASPIRATION_WINDOW = 50   # centipawns
MAX_RETRIES = 4
LMR_MOVE_THRESHOLD = 3   # reduce moves after this index
LMR_MIN_DEPTH = 3        # minimum depth to apply LMR
NULL_MOVE_REDUCTION = 2   # R value (usually 2 or 3)
NULL_MOVE_MIN_DEPTH = 3

TTEntry = namedtuple("TTEntry", ["depth", "score", "flag"])
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

transposition_table = {}
material_eval_table = {}
dnn_eval_table = {}
killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
history_heuristic = {}
dnn_evals = 0

kpi = {
    "mat_eval": 0,
    "beta_cutoff": 0,
    "tt_hits": 0,
    "met_hits": 0,
    "det_hits": 0,
    "q_depth": 0,
}
# ----------------------------------
# Piece Values
# ----------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def evaluate_material(board: chess.Board) -> int:
    """
    Simple material evaluation from side-to-move perspective
    """
    if board.is_checkmate():
        return -INF
    if board.is_stalemate():
        return 0

    key = chess.polyglot.zobrist_hash(board)
    if key in material_eval_table:
        kpi['met_hits'] += 1
        return material_eval_table[key]

    kpi['mat_eval'] += 1
    score = get_material_eval(board)

    if len(material_eval_table) > MAX_ET_SIZE:
        material_eval_table.clear()
    material_eval_table[key] = score

    return score


def evaluate_dnn(board: chess.Board) -> int:
    global dnn_evals

    if board.is_checkmate():
        return -INF
    if board.is_stalemate():
        return 0

    key = chess.polyglot.zobrist_hash(board)
    if key in dnn_eval_table:
        kpi['det_hits'] += 1
        return dnn_eval_table[key]

    assert(not is_capture(board))
    dnn_evals += 1
    score = dnn_evaluation(board)
    if not board.turn: # black's move
        score = -score

    if len(dnn_eval_table) > MAX_ET_SIZE:
        dnn_eval_table.clear()
    dnn_eval_table[key] = score

    return int(score)


def move_score(board, move, depth):
    score = 0

    # Killer moves (quiet moves only)
    if not board.is_capture(move) and depth is not None:
        if move == killer_moves[depth][0]:
            score += 9000
        elif move == killer_moves[depth][1]:
            score += 8000

    score += history_heuristic.get(
        (move.from_square, move.to_square), 0
    )

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]

    if board.gives_check(move):
        score += 50

    return score

def ordered_moves(board, depth, pv_move=None):
    moves = list(board.legal_moves)

    if pv_move and pv_move in moves:
        moves.remove(pv_move)

    moves.sort(key=lambda m: move_score(board, m, depth), reverse=True)

    if pv_move:
        return [pv_move] + moves
    return moves


def quiescence(board, alpha, beta, q_depth):
    if q_depth > kpi['q_depth']:
        kpi['q_depth'] = q_depth

    stand_pat = evaluate_material(board)

    if dnn_evals < MAX_DNN_EAVALS and q_depth <= DNN_MAX_Q_DEPTH and not is_capture(board) and \
        abs(stand_pat - beta) < DNN_SCORE_DIFF_THRESH:
        stand_pat = evaluate_dnn(board)

    if stand_pat >= beta:
        kpi['beta_cutoff'] += 1
        return beta

    if  dnn_evals < MAX_DNN_EAVALS and q_depth <= DNN_MAX_Q_DEPTH and not is_capture(board) and \
            (abs(stand_pat - alpha) < DNN_SCORE_DIFF_THRESH or alpha < stand_pat):
        stand_pat = evaluate_dnn(board)

    if alpha < stand_pat:
        alpha = stand_pat

    for move in ordered_moves(board, depth=None): #board.legal_moves:
        if q_depth <= TACTICAL_Q_DEPTH:
            if not ((board.is_check() or board.is_capture(move) or move.promotion or
                     board.gives_check(move))):
                continue
        else:
            if not board.is_capture(move):
                continue

        board.push(move)
        score = -quiescence(board, -beta, -alpha, q_depth + 1)
        board.pop()

        if score >= beta:
            kpi['beta_cutoff'] += 1
            return beta
        if score > alpha:
            alpha = score

    return alpha

def negamax(board, depth, alpha, beta):
    alpha_orig = alpha
    key = chess.polyglot.zobrist_hash(board)

    # -------- Transposition Table --------
    if key in transposition_table:
        kpi['tt_hits'] += 1
        entry = transposition_table[key]
        if entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score
            elif entry.flag == LOWERBOUND:
                alpha = max(alpha, entry.score)
            elif entry.flag == UPPERBOUND:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

    # -------- Leaf --------
    if depth == 0:
        return quiescence(board, alpha, beta, 1)

    in_check = board.is_check()
    max_eval = -INF

    # =====================================
    # Null Move Pruning
    # =====================================
    if (
        depth >= NULL_MOVE_MIN_DEPTH
        and not in_check
        and not board.is_checkmate()
    ):
        board.push(chess.Move.null())
        score = -negamax(
            board,
            depth - 1 - NULL_MOVE_REDUCTION,
            -beta,
            -beta + 1
        )
        board.pop()

        # -------- Fail-high → prune --------
        if score >= beta:
            return beta

    # =====================================
    # Normal Move Search (with LMR)
    # =====================================
    moves = ordered_moves(board, depth)

    for move_index, move in enumerate(moves):
        board.push(move)

        # -------- LMR Decision --------
        reduce = (
            depth >= LMR_MIN_DEPTH
            and move_index >= LMR_MOVE_THRESHOLD
            and not in_check
            and not board.is_capture(move)
            and not board.gives_check(move)
            and move != killer_moves[depth][0]
            and move != killer_moves[depth][1]
        )

        if reduce:
            score = -negamax(
                board,
                depth - 2,
                -alpha - 1,
                -alpha
            )
            if score > alpha:
                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha
                )
        else:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha
            )

        board.pop()

        if score > max_eval:
            max_eval = score

        if score > alpha:
            alpha = score

        # -------- Beta Cutoff --------
        if alpha >= beta:
            if not board.is_capture(move):
                # Killer update
                if killer_moves[depth][0] != move:
                    killer_moves[depth][1] = killer_moves[depth][0]
                    killer_moves[depth][0] = move

                # History update
                key_hist = (move.from_square, move.to_square)
                history_heuristic[key_hist] = history_heuristic.get(key_hist, 0) + depth * depth
            break

    # -------- TT Store --------
    if max_eval <= alpha_orig:
        flag = UPPERBOUND
    elif max_eval >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT

    if len(transposition_table) > MAX_TT_SIZE:
        kpi['tt_clears'] += 1
        transposition_table.clear()
    transposition_table[key] = TTEntry(depth, max_eval, flag)
    return max_eval


def age_history():
    for k in history_heuristic:
        history_heuristic[k] //= 2


def find_best_move(fen, max_depth=MAX_DEPTH):
    global dnn_evals, killer_moves, history_heuristic
    dnn_evals = 0
    killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
    history_heuristic = {}

    board = chess.Board(fen)
    best_move = None
    best_score = 0
    pv_move = None

    for depth in range(1, max_depth + 1):
        age_history()

        window = ASPIRATION_WINDOW
        retries = 0

        while True:
            alpha = best_score - window
            beta = best_score + window
            alpha_orig = alpha

            current_best_move = None
            current_best_score = -INF

            for move in ordered_moves(board, depth, pv_move):
                board.push(move)
                score = -negamax(board, depth - 1, -beta, -alpha)
                board.pop()

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move

                if score > alpha:
                    alpha = score

            # -------- SUCCESS --------
            if current_best_score > alpha_orig and current_best_score < beta:
                best_move = current_best_move
                best_score = current_best_score
                pv_move = best_move
                break

            # -------- FAIL-LOW --------
            if current_best_score <= alpha_orig:
                window *= 2

            # -------- FAIL-HIGH --------
            elif current_best_score >= beta:
                window *= 2

            retries += 1

            # -------- FALLBACK --------
            if retries >= MAX_RETRIES:
                alpha = -INF
                beta = INF

                current_best_score = -INF
                for move in ordered_moves(board, depth, pv_move):
                    board.push(move)
                    score = -negamax(board, depth - 1, -beta, -alpha)
                    board.pop()

                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move

                    if score > alpha:
                        alpha = score

                best_move = current_best_move
                best_score = current_best_score
                pv_move = best_move
                break

        print(f"Depth {depth}: Best={best_move}, Score={best_score}")

    return best_move, best_score


def  main():

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break

            for key in kpi:
                kpi[key] = 0

            start_time = time.perf_counter()
            move, score = find_best_move(fen)
            end_time = time.perf_counter()

            kpi['dnn_evals'] = dnn_evals
            kpi['tt_size'] = len(transposition_table)
            kpi['met_size'] = len(material_eval_table)
            kpi['det_size'] = len(dnn_eval_table)
            kpi['time'] = int(end_time - start_time)
            print(kpi)
            print(f"Best move: {move}")
            print(f"Evaluation: {score}")

        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break

if __name__ == '__main__':
    main()

# 2r2rk1/1p1nppbp/p1npb1p1/q7/N1P1P3/4BP2/PP2B1PP/2RQNRK1 b - - 5 14

# {'dnn_eval': 300, 'mat_eval': 928971, 'beta_cutoff': 841436, 'tt_hits': 2084, 'met_hits': 203026, 'det_hits': 28,
# 'q_depth': 20, 'tt_size': 15424, 'met_size': 128967, 'det_size': 300, 'time': 356}

# {'dnn_eval': 300, 'mat_eval': 456811, 'beta_cutoff': 427040, 'tt_hits': 1590, 'met_hits': 130554, 'det_hits': 157,
# 'q_depth': 20, 'tt_size': 7185, 'met_size': 56809, 'det_size': 300, 'time': 186} - Iterative deepening

# {'mat_eval': 178543, 'beta_cutoff': 162776, 'tt_hits': 1203, 'met_hits': 42916, 'det_hits': 90, 'q_depth': 21,
# 'dnn_evals': 300, 'tt_size': 4053, 'met_size': 178543, 'det_size': 300, 'time': 79} - Killer moves

# {'mat_eval': 160659, 'beta_cutoff': 149451, 'tt_hits': 1087, 'met_hits': 42584, 'det_hits': 149, 'q_depth': 21,
# 'dnn_evals': 300, 'tt_size': 3737, 'met_size': 160659, 'det_size': 300, 'time': 74} - Heuristic

# {'mat_eval': 118717, 'beta_cutoff': 111007, 'tt_hits': 691, 'met_hits': 30916, 'det_hits': 133, 'q_depth': 21,
# 'dnn_evals': 300, 'tt_size': 3309, 'met_size': 118717, 'det_size': 300, 'time': 60} - Aspiration window

# {'mat_eval': 116721, 'beta_cutoff': 111522, 'tt_hits': 722, 'met_hits': 33994, 'det_hits': 133, 'q_depth': 21,
# 'dnn_evals': 300, 'tt_size': 3508, 'met_size': 116721, 'det_size': 300, 'time': 62}- LMR

# {'mat_eval': 96027, 'beta_cutoff': 89159, 'tt_hits': 352, 'met_hits': 23022, 'det_hits': 133, 'q_depth': 20,
# 'dnn_evals': 300, 'tt_size': 2799, 'met_size': 96027, 'det_size': 300, 'time': 49} _ Null move pruning
