from typing import Any

import numpy as np
import tensorflow as tf
from numpy import dtype, ndarray
from numpy.typing import NDArray
from typing import Any
from build_model import tanh_to_score
from cached_board import CachedBoard

INF = 10_000

class DNNEvaluator:
    """DNN-based position evaluator with model caching."""

    def __init__(self):
        self._model = None
        self._model_path = None

    def _load_model(self, filepath: str):
        """Load model if not already loaded or if path changed."""
        if self._model is None or self._model_path != filepath:
            self._model = tf.keras.models.load_model(filepath)
            self._model.summary()
            self._model_path = filepath

    def evaluate(self, board: CachedBoard, model_filepath) -> int:
        """
        Evaluate position from side-to-move perspective.

        Returns:
            Score in centipawns. Positive = good for side to move.
        """
        # Load model if needed
        self._load_model(model_filepath)

        # Get board representation and predict
        board_repr = board.get_board_repr()
        board_repr = np.expand_dims(board_repr, axis=0)

        return self.dnn_eval_board_repr(board_repr)

    def dnn_eval_board_repr(self, board_repr: NDArray[Any]) -> int:
        # Direct call is MUCH faster for single samples
        score = self._model(board_repr, training=False)[0][0].numpy()
        score = tanh_to_score(score)
        return int(score)


# Global evaluator instance
_evaluator = DNNEvaluator()


def dnn_eval(board: CachedBoard, model_filepath) -> int:
    """
    Evaluate position using DNN model.

    Args:
        board: Position to evaluate (should be quiet position)
        model_filepath: Path to Keras model file

    Returns:
        Score in centipawns from side-to-move perspective
    """
    return _evaluator.evaluate(board, model_filepath)


def main():
    while True:
        try:
            fen = input("FEN: ").strip()
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen.lower() == "exit":
                break

            board = CachedBoard(fen)
            score = dnn_eval(board)
            print(f"Predicted score: {score} cp")
            print(f"Side to move: {'White' if board.turn else 'Black'}")

        except KeyboardInterrupt:
            if input("\nType 'exit' to exit: ").strip().lower() == "exit":
                break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()