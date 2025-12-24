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

MAX_DEPTH = 1
INF = 10_000
MAX_TT_SIZE = 200_000
MAX_ET_SIZE = 200_000

TTEntry = namedtuple("TTEntry", ["depth", "score", "flag"])
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

transposition_table = {}
evaluation_table = {}

kpi = {
    "dnn_eval": 0,
    "mat_eval": 0,
    "beta_cutoff": 0,
    "tt_hits": 0,
    "tt_clears": 0,
    "et_hits": 0,
    "et_clears": 0,
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

# ----------------------------------
# Evaluation Function
# ----------------------------------

def evaluate(board: chess.Board) -> int:
    """
    Simple material evaluation from side-to-move perspective
    """
    if board.is_checkmate():
        return -INF
    if board.is_stalemate():
        return 0

    key = chess.polyglot.zobrist_hash(board)
    if key in evaluation_table:
        kpi['et_hits'] += 1
        return evaluation_table[key]

    if not is_capture(board):
        kpi['dnn_eval'] += 1
        score = dnn_evaluation(board)
        if not board.turn: # black's move
            score = -score
    else:
        kpi['mat_eval'] += 1
        score = get_material_eval(board)

    if len(evaluation_table) > MAX_ET_SIZE:
        kpi['et_clears'] += 1
        evaluation_table.clear()
    evaluation_table[key] = score

    return score

# ----------------------------------
# Move Ordering
# ----------------------------------

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

def ordered_moves(board):
    return sorted(board.legal_moves, key=lambda m: move_score(board, m), reverse=True)

# ----------------------------------
# Quiescence Search
# ----------------------------------

def quiescence(board, alpha, beta, q_depth):
    if q_depth > kpi['q_depth']:
        kpi['q_depth'] = q_depth

    stand_pat = evaluate(board)
    # if abs(stand_pat - beta) < DNN_THRESH and  not is_capture(board)
        # kpi['dnn_eval'] += 1
        # stand_pat = score = dnn_evaluation(board)
    #         if not board.turn: # black's move
    #             score = -score
    if stand_pat >= beta:
        kpi['beta_cutoff'] += 1
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move): # Todo for first 'n' moves consider check
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

def find_best_move(fen, depth=MAX_DEPTH):
    board = chess.Board(fen)
    best_move = None
    best_score = -INF

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -INF, INF)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

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
            kpi['et_size'] = len(evaluation_table)
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

# 2r2rk1/1p1nppbp/p1npb1p1/q7/N1P1P3/4BP2/PPN1B1PP/2RQ1RK1 w - - 4 14

# {'dnn_eval': 23909, 'mat_eval': 142743, 'beta_cutoff': 256823, 'tt_hits': 0, 'tt_clears': 0,
# 'et_hits': 231429, 'et_clears': 0, 'q_depth': 26, 'tt_size': 0, 'et_size': 166652,
# 'time': 1031} - DEPTH=1
