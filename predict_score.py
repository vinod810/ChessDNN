import sys
import chess.pgn
import numpy as np
import tensorflow as tf
from chess import Board

from build_model import MODEL_FILEPATH, TANH_FACTOR, tanh_to_score
from prepare_data import get_board_repr


def  main():
    #model = tf.keras.models.load_model(MODEL_FILEPATH)
    #model.summary()

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


def dnn_evaluation(board: Board) -> float:

    if board.is_checkmate():
        return -10000 if board.turn else 10000

    if dnn_evaluation.model is None:
        dnn_evaluation.model = tf.keras.models.load_model(MODEL_FILEPATH)
        dnn_evaluation.model.summary()

    board_repr = get_board_repr(board)
    board_repr = np.expand_dims(board_repr, axis=0)  # batch size = 1
    score = dnn_evaluation.model.predict(board_repr, verbose=0)[0][0]
    score = tanh_to_score(score)  # round((score[0][0] * 2.0 * Max_Score - Max_Score) / 100, 1)
    #if board.turn == chess.BLACK:
    #    score = -score
    return score
dnn_evaluation.model = None


if __name__ == '__main__':
    main()