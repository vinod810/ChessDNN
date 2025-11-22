import chess
import math
import tensorflow as tf

from build_model import MODEL_FILEPATH
from predict_score import evaluate_position
from prepare_data import MAX_SCORE


def score_move(board, move):
    """
    Heuristic score for move ordering:
    - Captures: MVV-LVA
    - Promotions: high priority
    - Checks: moderate priority
    - Quiet moves: low priority
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        # MVV-LVA: victim value * 10 - attacker value
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        v_val = piece_values[victim.piece_type] if victim else 0
        a_val = piece_values[attacker.piece_type] if attacker else 0
        score += 10 * v_val - a_val + 10000  # captures get very high score
    if move.promotion:
        score += 9000
    if board.gives_check(move):
        score += 500
    return score

def ordered_moves(board):
    """
    Returns a list of legal moves ordered by heuristic score.
    """
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m), reverse=True)
    return moves

def quiescence(board, alpha, beta, evaluate, model, qdepth, max_qdepth=3):
    stand_pat = evaluate(board, model)
    #if board.turn == chess.BLACK:
    #    stand_pat = -stand_pat
    move = board.pop()
    board.push(move)
    print("quiescence: ", stand_pat, move, qdepth, alpha, beta)

    if board.is_checkmate():
        # Assign a large negative value for checkmate for the current player
        # Note: This is a simplification; handling checkmate correctly in negamax requires care.
        return -MAX_SCORE, []

    if stand_pat >= beta:
        print("bete return")
        return beta, []

    if alpha < stand_pat:
        alpha = stand_pat

    if qdepth >= max_qdepth:
        return stand_pat, []

    best_pv = []
    for move in ordered_moves(board):
        if board.is_capture(move) or move.promotion or board.gives_check(move):
            board.push(move)
            score, child_pv = quiescence(board, -beta, -alpha, evaluate, model, qdepth + 1, max_qdepth)
            score = -score
            board.pop()

            if score >= beta:
                return beta, [board.san(move)]
            if score > alpha:
                alpha = score
                best_pv = [board.san(move)] + child_pv
    return alpha, best_pv

def alpha_beta(board, depth, alpha, beta, evaluate, model, max_qdepth=3):
    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, evaluate, model, qdepth=0, max_qdepth=max_qdepth)

    max_score = -math.inf
    best_pv = []
    for move in ordered_moves(board):
        board.push(move)
        score, child_pv = alpha_beta(board, depth - 1, -beta, -alpha, evaluate, model, max_qdepth)
        score = -score
        board.pop()

        if score >= beta:
            return score, [board.san(move)]
        if score > max_score:
            max_score = score
            best_pv = [board.san(move)] + child_pv
        if score > alpha:
            alpha = score
    return max_score, best_pv

def find_best_move(board, depth, evaluate, model, max_qdepth=3):
    best_move = None
    best_score = -math.inf
    best_pv = []
    for move in ordered_moves(board):
        board.push(move)
        print("find_best_move", move)
        score, pv = alpha_beta(board, depth - 1, -math.inf, math.inf, evaluate, model, max_qdepth)
        print(f"find_best_move: score={score}, pv={score}")
        #print(score, move)
        score = -score
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
            best_pv = [board.san(move)] + pv
    return best_move, best_score, best_pv

def main():
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    model.summary()

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break
            board = chess.Board(fen)
            best_move, score, pv_san = find_best_move(board, depth=2, evaluate=evaluate_position, model=model, max_qdepth=6)
            print("Best move:", board.san(best_move))
            print("Evaluation:", score)
            print("PV:", " -> ".join(pv_san))
            print("predicted score: ", score)
        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break

    board = chess.Board()

if __name__ == '__main__':
    main()
