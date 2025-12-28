import time
import chess
import chess.polyglot
from collections import namedtuple

from position_eval import positional_eval # Returns positional evaluation using piece tables.
from dnn_eval import dnn_eval # Returns positional evaluation using a DNN model.
from prepare_data import is_any_move_capture

INF = 10_000
MAX_NEGAMAX_DEPTH = 20
MAX_TIME = 30
MAX_TABLE_SIZE = 200_000

DELTA_MAX_DNN_EVAL = 50 # Score difference, below which will trigger a DNM evaluation
STAND_PAT_MAX_DNN_EVAL = 200
IS_MATERIAL_ONLY_EVAL = False
TACTICAL_QS_MAX_DEPTH = 3 # After this quiescent search depth, only captures are considered, i.e. no checks or promotions.
ASPIRATION_WINDOW = 40
MAX_AW_RETRIES = 3
LMR_MOVE_THRESHOLD = 3   # reduce moves after this index
LMR_MIN_DEPTH = 3        # minimum depth to apply LMR
NULL_MOVE_REDUCTION = 2   # R value (usually 2 or 3)
NULL_MOVE_MIN_DEPTH = 3
DELTA_PRUNING_QS_MIN_DEPTH = 10
DELTA_PRUNING_MARGIN = 50

class TimeControl:
    time_limit = None  # in seconds
    start_time = None
    stop_search = False

TTEntry = namedtuple("TTEntry", ["depth", "score", "flag", "best_move"])
TT_EXACT, TT_LOWER_BOUND, TT_UPPER_BOUND = 0, 1, 2

transposition_table = {}
qs_transposition_table = {}
pos_eval_cache = {}
dnn_eval_cache = {}
killer_moves = [[None, None] for _ in range(MAX_NEGAMAX_DEPTH + 1)]
history_heuristic = {}

kpi = {
    "pos_eval": 0,
    "dnn_evals": 0,
    "beta_cutoffs": 0,
    "tt_hits": 0,
    "qs_tt_hits": 0,
    "pec_hits": 0, # positional evaluation cache hits
    "dec_hits": 0,  # DNN evaluation cache hits
    "q_depth": 0,
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}


def check_time():
    if TimeControl.time_limit is None:
        return

    if (time.perf_counter() - TimeControl.start_time) >= TimeControl.time_limit:
        TimeControl.stop_search = True


def evaluate_positional(board: chess.Board) -> int:
    key = chess.polyglot.zobrist_hash(board)
    if key in pos_eval_cache:
        kpi['pec_hits'] += 1
        return pos_eval_cache[key]

    kpi['pos_eval'] += 1
    score = positional_eval(board, IS_MATERIAL_ONLY_EVAL)
    pos_eval_cache[key] = score
    return score


def evaluate_dnn(board: chess.Board) -> int:
    key = chess.polyglot.zobrist_hash(board)
    if key in dnn_eval_cache:
        kpi['dec_hits'] += 1
        return dnn_eval_cache[key]

    assert(not is_any_move_capture(board)) # DNN is trained for positions without captures.
    kpi['dnn_evals'] += 1
    score = int(dnn_eval(board))

    dnn_eval_cache[key] = score
    return score


def move_score_q_search(board: chess.Board, move) -> int:
    score = 0

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]

    if board.gives_check(move):
        score += 50

    return score


def move_score(board, move, depth):
    score = 0

    # Killer moves (quiet moves only)
    if not board.is_capture(move) and depth is not None:
        if move == killer_moves[depth][0]:
            score += 9000
        elif move == killer_moves[depth][1]:
            score += 8000

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]

    score += history_heuristic.get(
        (move.from_square, move.to_square), 0
    )

    if board.gives_check(move):
        score += 50

    return score


def ordered_moves(board, depth, pv_move=None, tt_move=None):
    moves = list(board.legal_moves)

    # 1. TT move first
    if tt_move and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)

    # 2. PV move second (if different)
    if pv_move and pv_move in moves:
        moves.remove(pv_move)
        moves.insert(0, pv_move)

    # 3. Sort remaining moves
    start = 0
    if pv_move:
        start += 1
    if tt_move and tt_move != pv_move:
        start += 1

    moves[start:] = sorted(
        moves[start:],
        key=lambda m: move_score(board, m, depth),
        reverse=True
    )

    return moves


def ordered_moves_q_search(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: move_score_q_search(board, m), reverse=True)
    return moves


def quiescence(board, alpha, beta, q_depth):
    if q_depth > kpi['q_depth']:
        kpi['q_depth'] = q_depth

    check_time()
    if TimeControl.stop_search:
        raise TimeoutError()

    key = chess.polyglot.zobrist_hash(board)
    if key in qs_transposition_table:
        kpi['qs_tt_hits'] += 1
        return qs_transposition_table[key]

    # Call positional evaluation first to reduce computing. We will call DNN evaluation later if necessary.
    stand_pat = evaluate_positional(board)

    # Call DNN evaluation when the scores are within a threshold. Note: DNN is trained only for positions without
    # captures.
    is_dnn_eval = False
    if (abs(stand_pat) < STAND_PAT_MAX_DNN_EVAL and abs(stand_pat - beta) < DELTA_MAX_DNN_EVAL and
            not is_any_move_capture(board)):
        stand_pat = evaluate_dnn(board)
        is_dnn_eval = True

    if stand_pat >= beta:
        kpi['beta_cutoffs'] += 1
        return beta

    # If already a queen down, no capture will improve the situation
    if stand_pat + PIECE_VALUES[chess.QUEEN] < alpha:
        return alpha

    # Call DNN evaluation when the scores are within a threshold or using evaluation as alpha
    if (not is_dnn_eval and abs(stand_pat) < STAND_PAT_MAX_DNN_EVAL and
            (abs(stand_pat - alpha) < DELTA_MAX_DNN_EVAL or alpha < stand_pat) and not is_any_move_capture(board)):
        stand_pat = evaluate_dnn(board)

    if alpha < stand_pat:
        alpha = stand_pat

    is_check = board.is_check()
    for move in ordered_moves_q_search(board): #board.legal_moves:
        if is_check:
            pass  # allow all legal moves
        elif not board.is_capture(move):
            if q_depth > TACTICAL_QS_MAX_DEPTH:
                continue
            if not board.gives_check(move):
                continue

        # -------- Simple delta pruning --------
        if board.is_capture(move) and q_depth >= DELTA_PRUNING_QS_MIN_DEPTH:
            victim = board.piece_at(move.to_square)
            if victim: # Not en-passant
                attacker = board.piece_at(move.from_square)
                gain = PIECE_VALUES[victim.piece_type] - (PIECE_VALUES[attacker.piece_type] if attacker else 0)
                if stand_pat + gain + DELTA_PRUNING_MARGIN < alpha:
                    continue

        board.push(move)
        score = -quiescence(board, -beta, -alpha, q_depth + 1)
        board.pop()

        if score >= beta:
            kpi['beta_cutoffs'] += 1
            return beta

        if score > alpha:
            alpha = score

    qs_transposition_table[key] = alpha
    return alpha


def has_non_pawn_material(board):
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        if board.pieces(piece_type, board.turn):
            return True
    return False


def negamax(board, depth, alpha, beta):
    alpha_orig = alpha
    key = chess.polyglot.zobrist_hash(board)
    best_move = None

    check_time()
    if TimeControl.stop_search:
        raise TimeoutError()

    # -------- Transposition Table --------
    if key in transposition_table:
        kpi['tt_hits'] += 1
        entry = transposition_table[key]
        if entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return entry.score
            elif entry.flag == TT_LOWER_BOUND:
                alpha = max(alpha, entry.score)
            elif entry.flag == TT_UPPER_BOUND:
                beta = min(beta, entry.score)
            if alpha >= beta:
                if entry.flag == TT_LOWER_BOUND:
                    return alpha
                elif entry.flag == TT_UPPER_BOUND:
                    return beta
                else:
                    return entry.score

    if depth == 0:
        return quiescence(board, alpha, beta, 1)

    in_check = board.is_check()
    max_eval = -INF

    # Null Move Pruning
    if (depth >= NULL_MOVE_MIN_DEPTH and not in_check and has_non_pawn_material(board) and
            board.occupied.bit_count() > 6):
        board.push(chess.Move.null())
        score = -negamax(board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1)
        board.pop()

        if score >= beta:
            return beta

    # Normal Move Search (with LMR)
    entry = transposition_table.get(key)
    tt_move = entry.best_move if entry else None
    moves = ordered_moves(board, depth, tt_move=tt_move)

    for move_index, move in enumerate(moves):
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        board.push(move)
        child_in_check = board.is_check()

        # -------- LMR Decision --------
        is_pv = (move_index == 0)
        reduce = (
                depth >= LMR_MIN_DEPTH
                and move_index >= LMR_MOVE_THRESHOLD
                and not is_pv
                and not child_in_check
                and not is_capture
                and not gives_check
                and move != killer_moves[depth][0]
                and move != killer_moves[depth][1]
        )

        if reduce:
            reduction = 1
            if depth >= 6 and move_index >= 6:
                reduction = 2
            score = -negamax(
                board,
                depth - reduction,
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
            best_move = move

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
                history_heuristic[key_hist] = min(
                    history_heuristic.get(key_hist, 0) + depth * depth,
                    10_000
                )

            break

    # -------- TT Store --------
    if max_eval <= alpha_orig:
        flag = TT_UPPER_BOUND
    elif max_eval >= beta:
        flag = TT_LOWER_BOUND
    else:
        flag = TT_EXACT

    old = transposition_table.get(key)
    if old is None or depth >= old.depth:
        transposition_table[key] = TTEntry(depth, max_eval, flag, best_move)

    return max_eval


def age_heuristic_history():
    for k in list(history_heuristic.keys()):
        history_heuristic[k] = history_heuristic[k] * 3 // 4


def control_dict_size(table, max_dict_size):
    if len(table) > max_dict_size:
        for _ in range(max_dict_size // 4):
            table.pop(next(iter(table)))


def find_best_move(fen, max_depth=MAX_NEGAMAX_DEPTH, time_limit=None):
    TimeControl.time_limit = time_limit
    TimeControl.stop_search = False
    TimeControl.start_time = time.perf_counter()

    for i in range(len(killer_moves)):
        killer_moves[i] = [None, None]
    history_heuristic.clear()
    transposition_table.clear()
    qs_transposition_table.clear()

    control_dict_size(pos_eval_cache, MAX_TABLE_SIZE)
    control_dict_size(dnn_eval_cache, MAX_TABLE_SIZE)
    control_dict_size(qs_transposition_table, MAX_TABLE_SIZE)

    board = chess.Board(fen)
    best_move = None
    best_score = 0
    pv_move = None

    try:
        for depth in range(1, max_depth + 1):
            #check_time()
            #if TimeControl.stop_search:
            #    break  # exit if time is up

            #if (time.perf_counter() - TimeControl.start_time) * 4 > TimeControl.time_limit:
                # Unlikely to complete the next depth iteration
             #   break

            check_time()
            if TimeControl.stop_search:
                break

            age_heuristic_history()

            root_key = chess.polyglot.zobrist_hash(board)
            entry = transposition_table.get(root_key)
            tt_move = entry.best_move if entry and entry.best_move in board.legal_moves else None

            window = ASPIRATION_WINDOW
            retries = 0

            while True:
                alpha = best_score - window
                beta = best_score + window
                alpha_orig = alpha

                current_best_move = None
                current_best_score = -INF

                for i, move in enumerate(ordered_moves(board, depth, pv_move, tt_move)):
                    check_time()
                    if TimeControl.stop_search:
                        break

                    board.push(move)

                    if i == 0:
                        score = -negamax(board, depth - 1, -beta, -alpha)
                    else:
                        score = -negamax(board, depth - 1, -alpha - 1, -alpha)
                        if score > alpha:
                            score = -negamax(board, depth - 1, -beta, -alpha)

                    board.pop()

                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move

                    if score > alpha:
                        alpha = score

                    if alpha >= beta:
                        break

                # -------- SUCCESS --------
                if alpha_orig < current_best_score < beta:
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
                if retries >= MAX_AW_RETRIES:
                    alpha = -INF
                    beta = INF

                    current_best_score = -INF
                    for move in ordered_moves(board, depth, pv_move, tt_move):
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

    except TimeoutError:
        # Time up — return last completed depth's best move
        pass

    if best_move is None:
        # fallback if nothing searched
        best_move = list(board.legal_moves)[0]
        best_score = evaluate_positional(board)

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
            move, score = find_best_move(fen, max_depth=20, time_limit=60)
            end_time = time.perf_counter()

            kpi['tt_size'] = len(transposition_table)
            kpi['mec_size'] = len(pos_eval_cache)
            kpi['dec_size'] = len(dnn_eval_cache)
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

