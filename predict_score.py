import chess.pgn
import numpy as np
import tensorflow as tf
from chess import Board

from build_model import MODEL_FILEPATH, tanh_to_score
from prepare_data import get_board_repr, MAX_SCORE

def dnn_evaluation(board: Board) -> int:

    if board.is_checkmate():
        return -MAX_SCORE if board.turn else MAX_SCORE

    if dnn_evaluation.model is None:
        dnn_evaluation.model = tf.keras.models.load_model(MODEL_FILEPATH)
        dnn_evaluation.model.summary()

    board_repr = get_board_repr(board)
    board_repr = np.expand_dims(board_repr, axis=0)

    score = dnn_evaluation.model.predict(board_repr, verbose=0)[0][0]
    score = tanh_to_score(score)
    if not board.turn: # black's turn
        score = -score
    return int(score)

dnn_evaluation.model = None


def  main():

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break
            board = chess.Board(fen)
            score = dnn_evaluation(board)
            print("predicted score: ", score)
        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break


if __name__ == '__main__':
    main()