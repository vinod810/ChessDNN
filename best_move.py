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

TTEntry = namedtuple("TTEntry", ["depth", "score", "flag"])
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

transposition_table = {}
material_eval_table = {}
dnn_eval_table = {}

kpi = {
    "dnn_eval": 0,
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

    if board.is_checkmate():
        return -INF
    if board.is_stalemate():
        return 0

    key = chess.polyglot.zobrist_hash(board)
    if key in dnn_eval_table:
        kpi['det_hits'] += 1
        return dnn_eval_table[key]

    assert(not is_capture(board))
    kpi['dnn_eval'] += 1
    score = dnn_evaluation(board)
    if not board.turn: # black's move
        score = -score

    if len(dnn_eval_table) > MAX_ET_SIZE:
        dnn_eval_table.clear()
    dnn_eval_table[key] = score

    return int(score)


def move_score(board, move):
    """
    Heuristic for move ordering:
    Captures > Checks > Quiet moves
    """
    score = 0

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]

    if board.gives_check(move):
        score += 50

    return score

def ordered_moves(board, pv_move=None):
    moves = list(board.legal_moves)

    if pv_move and pv_move in moves:
        moves.remove(pv_move)
        moves.sort(key=lambda m: move_score(board, m), reverse=True)
        return [pv_move] + moves

    return sorted(moves, key=lambda m: move_score(board, m), reverse=True)


def quiescence(board, alpha, beta, q_depth):
    if q_depth > kpi['q_depth']:
        kpi['q_depth'] = q_depth

    stand_pat = evaluate_material(board)

    if kpi['dnn_eval'] < MAX_DNN_EAVALS and q_depth <= DNN_MAX_Q_DEPTH and not is_capture(board) and \
        abs(stand_pat - beta) < DNN_SCORE_DIFF_THRESH:
        stand_pat = evaluate_dnn(board)

    if stand_pat >= beta:
        kpi['beta_cutoff'] += 1
        return beta

    if  kpi['dnn_eval'] < MAX_DNN_EAVALS and q_depth <= DNN_MAX_Q_DEPTH and not is_capture(board) and \
            (abs(stand_pat - alpha) < DNN_SCORE_DIFF_THRESH or alpha < stand_pat):
        stand_pat = evaluate_dnn(board)

    if alpha < stand_pat:
        alpha = stand_pat

    for move in ordered_moves(board): #board.legal_moves:
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

# ----------------------------------
# Negamax with Alpha-Beta + TT
# ----------------------------------

def negamax(board, depth, alpha, beta):
    alpha_orig = alpha
    key = chess.polyglot.zobrist_hash(board)

    # Transposition table lookup
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

    if depth == 0:
        return quiescence(board, alpha, beta, 1)

    max_eval = -INF

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > max_eval:
            max_eval = score

        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff

    # Store in TT
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

# ----------------------------------
# Root Search
# ----------------------------------

def find_best_move(fen, max_depth=MAX_DEPTH):
    board = chess.Board(fen)
    best_move = None
    best_score = -INF
    pv_move = None

    for depth in range(1, max_depth + 1):
        alpha = -INF
        beta = INF
        current_best_move = None
        current_best_score = -INF

        for move in ordered_moves(board, pv_move):
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > current_best_score:
                current_best_score = score
                current_best_move = move

            alpha = max(alpha, score)

        best_move = current_best_move
        best_score = current_best_score
        pv_move = best_move  # PV from this iteration

        print(f"Depth {depth}: Best move = {best_move}, Score = {best_score}")

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