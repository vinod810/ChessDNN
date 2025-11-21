import sys
import chess.pgn
import numpy as np
import tensorflow as tf

from build_model import MODEL_FILEPATH, TANH_FACTOR
from prepare_data import get_board_repr


def  main():
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
            board_repr = get_board_repr(board)
            board_repr = np.expand_dims(board_repr, axis=0)  # batch size = 1
            #score = model.predict([board_repr], steps=1)[0]
            score = model.predict(board_repr)[0][0]
            score = round(np.arctanh(score) * TANH_FACTOR)# round((score[0][0] * 2.0 * Max_Score - Max_Score) / 100, 1)
            print("predicted score: ", score)
        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break


if __name__ == '__main__':
    main()