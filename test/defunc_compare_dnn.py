"""
Compare performance of dnn_eval_board_repr() between TensorFlow and NumPy implementations.
"""

import time
import numpy as np
from chess import Board

from defunc_build_model import DNN_MODEL_FILEPATH, tanh_to_score
from defunc_prepare_data import get_board_repr_and_material
from cached_board import CachedBoard

# Import evaluators from their respective modules
from defunc_dnn_eval import DNNEvaluator
from defunc_dnn_eval_numpy import DNNEvaluatorNumPyOptimized


# Test FEN position
TEST_FEN = "r2q1rk1/1bpn1pbp/1p2pnp1/p2p4/3P3P/1P3NP1/PBPNPPB1/2RQ1RK1 w - - 0 1"

# Number of iterations for timing (excluding warmup)
NUM_ITERATIONS = 100


def main():
    print("=" * 60)
    print("DNN Evaluation Performance Comparison")
    print("=" * 60)
    print(f"\nTest FEN: {TEST_FEN}")
    print(f"Iterations for timing: {NUM_ITERATIONS}")

    # Prepare board representations for both implementations
    # TensorFlow version uses CachedBoard
    cached_board = CachedBoard(TEST_FEN)
    board_repr = cached_board.get_board_repr()
    board_repr = np.expand_dims(board_repr, axis=0)

    # =========================================================================
    # First invocation (includes model loading) - Compare outputs
    # =========================================================================
    print("\n" + "-" * 60)
    print("First Invocation (includes model loading)")
    print("-" * 60)

    # TensorFlow implementation - first call
    tf_evaluator = DNNEvaluator()
    tf_evaluator._load_model(DNN_MODEL_FILEPATH)
    tf_score_first = tf_evaluator.dnn_eval_board_repr(board_repr)

    # NumPy implementation - first call
    np_evaluator = DNNEvaluatorNumPyOptimized()
    np_evaluator._load_model(DNN_MODEL_FILEPATH)
    np_score_first = np_evaluator.dnn_eval_board_repr(board_repr)

    print(f"\nTensorFlow score: {tf_score_first}")
    print(f"NumPy score:      {np_score_first}")
    print(f"Difference:       {abs(tf_score_first - np_score_first)}")

    if tf_score_first == np_score_first:
        print("✓ Outputs match exactly!")
    else:
        print(f"⚠ Outputs differ by {abs(tf_score_first - np_score_first)} centipawns")

    # =========================================================================
    # Performance comparison (model already loaded)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Performance Comparison (model pre-loaded)")
    print("-" * 60)

    # Warm up (not counted)
    for _ in range(5):
        tf_evaluator.dnn_eval_board_repr(board_repr)
        np_evaluator.dnn_eval_board_repr(board_repr)

    # Time TensorFlow implementation
    tf_times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        tf_evaluator.dnn_eval_board_repr(board_repr)
        end = time.perf_counter()
        tf_times.append(end - start)

    # Time NumPy implementation
    np_times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        np_evaluator.dnn_eval_board_repr(board_repr)
        end = time.perf_counter()
        np_times.append(end - start)

    # Calculate statistics
    tf_mean = np.mean(tf_times) * 1000  # Convert to ms
    tf_std = np.std(tf_times) * 1000
    tf_min = np.min(tf_times) * 1000
    tf_max = np.max(tf_times) * 1000

    np_mean = np.mean(np_times) * 1000
    np_std = np.std(np_times) * 1000
    np_min = np.min(np_times) * 1000
    np_max = np.max(np_times) * 1000

    print("\nTensorFlow Implementation:")
    print(f"  Mean:   {tf_mean:.4f} ms")
    print(f"  Std:    {tf_std:.4f} ms")
    print(f"  Min:    {tf_min:.4f} ms")
    print(f"  Max:    {tf_max:.4f} ms")

    print("\nNumPy Implementation:")
    print(f"  Mean:   {np_mean:.4f} ms")
    print(f"  Std:    {np_std:.4f} ms")
    print(f"  Min:    {np_min:.4f} ms")
    print(f"  Max:    {np_max:.4f} ms")

    # Comparison
    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)

    speedup = tf_mean / np_mean if np_mean < tf_mean else np_mean / tf_mean
    faster = "NumPy" if np_mean < tf_mean else "TensorFlow"

    print(f"\n{faster} is {speedup:.2f}x faster")
    print(f"Time saved per call: {abs(tf_mean - np_mean):.4f} ms")


if __name__ == "__main__":
    main()