import chess.pgn
import numpy as np
import tensorflow as tf
from chess import Board

from build_model import DNN_MODEL_FILEPATH, tanh_to_score
from prepare_data import get_board_repr, MAX_SCORE


def dnn_eval(board: Board) -> int:

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return MAX_SCORE if board.turn == chess.WHITE else -MAX_SCORE

        elif result == "0-1":
            return -MAX_SCORE if board.turn == chess.WHITE else MAX_SCORE

        else:
            return 0  # Draw

    if dnn_eval.model is None:
        dnn_eval.model = tf.keras.models.load_model(DNN_MODEL_FILEPATH)
        dnn_eval.model.summary()

    board_repr = get_board_repr(board)
    board_repr = np.expand_dims(board_repr, axis=0)

    score = dnn_eval.model.predict(board_repr, verbose=0)[0][0]
    score = tanh_to_score(score)
    if not board.turn: # black's turn
        score = -score
    return int(score)

dnn_eval.model = None


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
            score = dnn_eval(board)
            print("predicted score: ", score)
        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break


if __name__ == '__main__':
    main()