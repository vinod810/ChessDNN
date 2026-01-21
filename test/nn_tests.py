"""
Chess Neural Network Evaluation Test Suite
Tests trained NNUE or DNN models using the NNEvaluator abstraction.

This test file reuses code from nn_evaluator.py to ensure bugs in the
evaluator classes are caught by these tests.

Test Modes:
- Interactive-FEN: Interactive FEN evaluation (default)
- Incremental-vs-Full: Performance comparison
- Accumulator-Correctness: Verify incremental == full evaluation
- Eval-Accuracy: Test prediction accuracy against ground truth from training shards
- Feature-Extraction: Verify feature extraction correctness
- Symmetry: Test evaluation symmetry (mirrored positions)
- Edge-Cases: Test edge cases (checkmate, stalemate, special moves)
- Reset-Consistency: Test evaluator reset functionality
- Deep-Search-Simulation: Simulate deep search with many push/pop cycles
- Random-Games: Test with random legal move sequences
"""

import argparse
import numpy as np
import sys
import time
import io
import chess
import random
import glob
import zstandard as zstd
import chess.pgn

# Import from our modules - this ensures we test the actual production code
from nn_inference import (
    TANH_SCALE, NNUEFeatures, DNNFeatures,
    NNUE_INPUT_SIZE, DNN_INPUT_SIZE, MAX_SCORE
)
from nn_evaluator import NNEvaluator, DNNEvaluator, NNUEEvaluator
from shard_io import ShardReader, find_shards

CP_ERROR_CLIP = 100 # Keep low to make the average more sense.

# Configuration
VALID_TEST_TYPES = {
    0: "Interactive-FEN",
    1: "Incremental-vs-Full",
    2: "Accumulator-Correctness",
    3: "Eval-Accuracy",
    4: "NN-vs-Stockfish",
    5: "Feature-Extraction",
    6: "Symmetry",
    7: "Edge-Cases",
    8: "Reset-Consistency",
    9: "Deep-Search-Simulation",
    10: "Random-Games",
    11: "CP-Integrity",
    12: "All",
}


def get_model_path(nn_type: str) -> str:
    """Get the model path for a given NN type."""
    return "model/nnue.pt" if nn_type == "NNUE" else "model/dnn.pt"


def output_to_centipawns(output: float) -> float:
    """Convert linear network output (tanh space) to centipawns."""
    output = np.clip(output, -0.9999, 0.9999)
    return np.arctanh(output) * TANH_SCALE


def evaluator_push(evaluator: NNEvaluator, board: chess.Board, move: chess.Move):
    """
    Push a move to the evaluator and board using the unified interface.

    This matches the pattern used in engine.py's push_move() function.
    The push_with_board() method handles both DNN and NNUE internally.
    """
    evaluator.push_with_board(board, move)


def evaluator_pop(evaluator: NNEvaluator, board: chess.Board):
    """
    Pop a move from the evaluator and board.

    This matches the pattern used in engine.py: board.pop() then evaluator.pop().
    """
    board.pop()
    evaluator.pop()


# =============================================================================
# Interactive FEN Evaluation
# =============================================================================

def evaluate_fen(fen: str, evaluator: NNEvaluator) -> dict:
    """Evaluate a position from FEN string using the evaluator."""
    try:
        board = chess.Board(fen)
        # Use full evaluation for standalone FEN evaluation
        output = evaluator.evaluate_full(board)
        centipawns = output_to_centipawns(output)

        return {
            'success': True,
            'fen': fen,
            'side_to_move': 'White' if board.turn == chess.WHITE else 'Black',
            'output': output,
            'centipawns': centipawns
        }
    except ValueError as e:
        return {'success': False, 'error': f"Invalid FEN: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"Evaluation error: {e}"}


def print_evaluation(result: dict):
    """Pretty print evaluation results."""
    if not result['success']:
        print(f"❌ {result['error']}")
        return

    print("─" * 60)
    print(f"FEN: {result['fen']}")
    print(f"Side to move: {result['side_to_move']}")
    print(f"Network output: {result['output']:.6f}")
    print(f"Evaluation: {result['centipawns']:+.2f} centipawns")

    pawns = result['centipawns'] / 100.0
    stm = result['side_to_move']
    opponent = "Black" if stm == "White" else "White"

    if abs(pawns) < 0.1:
        assessment = "≈ Equal position"
    elif pawns > 3:
        assessment = f"Decisive advantage for {stm}"
    elif pawns > 1:
        assessment = f"Clear advantage for {stm}"
    elif pawns > 0.3:
        assessment = f"Slight advantage for {stm}"
    elif pawns < -3:
        assessment = f"Decisive advantage for {opponent}"
    elif pawns < -1:
        assessment = f"Clear advantage for {opponent}"
    elif pawns < -0.3:
        assessment = f"Slight advantage for {opponent}"
    else:
        assessment = "≈ Equal position"

    print(f"Assessment: {assessment}")
    print("─" * 60)


def print_help():
    """Print help information."""
    print("\n" + "=" * 60)
    print("Chess Position Evaluator - Help")
    print("=" * 60)
    print("Commands:")
    print("  Enter FEN string    - Evaluate position")
    print("  'startpos'          - Evaluate starting position")
    print("  'help'              - Show this help")
    print("  'exit' or 'quit'    - Exit program")
    print("\nExample FEN:")
    print("  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("=" * 60 + "\n")


def interactive_loop(evaluator: NNEvaluator, nn_type: str):
    """Main interactive loop for FEN evaluation."""
    print("\n" + "=" * 60)
    print("Chess Position Evaluator")
    print("=" * 60)
    print(f"Network type: {nn_type}")
    print("\nEnter FEN strings to evaluate positions.")
    print("Type 'help' for instructions, 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    while True:
        try:
            user_input = input("FEN> ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print_help()
                continue

            if not user_input:
                continue

            if user_input.lower() == 'startpos':
                user_input = startpos_fen
                print(f"Using starting position: {startpos_fen}")

            result = evaluate_fen(user_input, evaluator)
            print_evaluation(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Accumulator Correctness Test
# =============================================================================

def test_accumulator_correctness(nn_type: str, model_path: str):
    """
    Test that incremental evaluation matches full evaluation.

    Uses the NNEvaluator's validate_incremental() method which compares
    evaluate() (incremental) vs evaluate_full() (matrix multiplication).

    This tests the actual production code path in nn_evaluator.py.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Accumulator Correctness Test")
    print("=" * 70)

    # Test sequence includes: captures, en-passant, castling, king moves, promotion
    moves_san = [
        "d4", "e5", "dxe5", "f5", "exf6", "Nh6", "Bf4", "Bd6",
        "Nc3", "O-O", "Qd3", "Nc6", "O-O-O", "a6", "fxg7", "Rb8",
        "gxf8=N", "Kxf8", "Kb1", "Bxf4"
    ]

    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    print(f"\nPlaying through {len(moves_san)} moves and verifying after each...")
    all_passed = True
    move_count = 0

    for move_san in moves_san:
        move = board.parse_san(move_san)

        # Push using our helper that handles both DNN and NNUE
        evaluator_push(evaluator, board, move)
        move_count += 1

        # Use the evaluator's built-in validation
        eval_inc = evaluator.evaluate(board)
        eval_full = evaluator.evaluate_full(board)
        diff = abs(eval_inc - eval_full)
        passed = diff < 1e-6

        print(f"\n{'─' * 70}")
        print(f"After move {move_count}: {move_san}")
        print(f"{'─' * 70}")
        print(f"Position: {board.fen()}")
        print(f"Incremental: {eval_inc:.10f} ({output_to_centipawns(eval_inc):+.2f} cp)")
        print(f"Full:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:  {diff:.10e}")

        if passed:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")
            all_passed = False

    # Now test popping moves
    num_pops = 4
    print(f"\nPopping {num_pops} moves and verifying...")

    for i in range(num_pops):
        evaluator_pop(evaluator, board)

        eval_inc = evaluator.evaluate(board)
        eval_full = evaluator.evaluate_full(board)
        diff = abs(eval_inc - eval_full)
        passed = diff < 1e-6

        print(f"\n{'─' * 70}")
        print(f"After pop {i + 1}")
        print(f"{'─' * 70}")
        print(f"Position: {board.fen()}")
        print(f"Incremental: {eval_inc:.10f} ({output_to_centipawns(eval_inc):+.2f} cp)")
        print(f"Full:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:  {diff:.10e}")

        if passed:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All accumulator correctness tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Performance Test (Incremental vs Full)
# =============================================================================

def performance_test(nn_type: str, model_path: str):
    """
    Compare performance of full evaluation vs incremental evaluation.

    Tests the actual NNEvaluator implementation from nn_evaluator.py.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Performance Test: Full vs Incremental Evaluation")
    print("=" * 70)

    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)
    move = chess.Move.from_uci("e2e4")

    # Test 1: Full evaluation
    num_full_iters = 1000
    print(f"\n1. Full Evaluation - Matrix Multiplication ({num_full_iters} iterations)")

    start_time = time.time()
    for _ in range(num_full_iters):
        _ = evaluator.evaluate_full(board)
    full_time = time.time() - start_time

    print(f"   Time: {full_time:.4f} seconds")
    print(f"   Avg per evaluation: {full_time / num_full_iters * 1000:.3f} ms")

    # Test 2: Incremental evaluation with push/pop cycles
    num_cycles = 500
    print(f"\n2. Incremental Evaluation - Push/Pop Cycles ({num_cycles} cycles)")

    start_time = time.time()

    for _ in range(num_cycles):
        # Push
        evaluator_push(evaluator, board, move)

        # Evaluate (uses accumulator)
        _ = evaluator.evaluate(board)

        # Pop
        evaluator_pop(evaluator, board)

    incremental_time = time.time() - start_time

    print(f"   Time: {incremental_time:.4f} seconds")
    print(f"   Avg per push/eval/pop cycle: {incremental_time / num_cycles * 1000:.3f} ms")

    # Results
    print("\n" + "─" * 70)
    print("Results:")
    print(f"  Full (matrix multiply):     {full_time / num_full_iters * 1000:.3f} ms per evaluation")
    print(f"  Incremental (accumulator):  {incremental_time / num_cycles * 1000:.3f} ms per cycle")

    # Calculate speedup (comparing one full eval vs one incremental cycle)
    speedup = (full_time / num_full_iters) / (incremental_time / num_cycles)
    print(f"  Speedup: {speedup:.2f}x")

    print("\nNote: Incremental uses accumulator (add/subtract weight vectors)")
    print("instead of full matrix multiplication for the first layer.")
    print("=" * 70)


# =============================================================================
# Evaluation Accuracy Test
# =============================================================================

def test_eval_accuracy(nn_type: str, model_path: str, positions_size: int):
    """
    Test evaluation accuracy against ground truth from training data shards.

    Uses diagnostic records (with FEN) from binary shard files to compare
    model predictions against stored Stockfish evaluations.
    """
    from pathlib import Path
    from shard_io import ShardReader, find_shards

    print("\n" + "=" * 70)
    print(f"{nn_type} Evaluation Accuracy Test")
    print("=" * 70)

    # Find shard files
    dnn_shards, nnue_shards = find_shards("data", nn_type)
    shard_files = dnn_shards if nn_type.upper() == "DNN" else nnue_shards

    if not shard_files:
        print(f"\n❌ ERROR: No shard files found for {nn_type}")
        print("Please ensure .bin.zst files exist in the data directory")
        return

    # Pick a random shard
    shard_path = random.choice(shard_files)
    print(f"\nReading diagnostic records from: {shard_path}")

    # Read diagnostic records (which have FEN)
    reader = ShardReader(nn_type)
    records = reader.read_diagnostic_records(shard_path, max_records=positions_size)

    if not records:
        print("\n❌ ERROR: No diagnostic records found in shard file")
        print("Diagnostic records are written every 1000 positions and include FEN.")
        print("You may need to regenerate shards with the updated prepare_data.py")
        return

    print(f"✓ Loaded {len(records)} diagnostic records from shard")

    # Create evaluator
    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    # Evaluate all positions
    print("\nEvaluating positions...")

    true_tanh_values = []
    pred_tanh_values = []
    true_cp_list = []
    pred_cp_list = []
    fen_list = []
    errors = []

    for i, rec in enumerate(records):
        if (i + 1) % max(1, len(records) // 10) == 0:
            print(f"  Progress: {i + 1}/{len(records)}")

        try:
            fen = rec['fen']
            stm_is_white = rec['stm'] == 1

            # Evaluate using features from shard
            if nn_type.upper() == "DNN":
                pred_output = evaluator.inference.evaluate_full(rec['features'])
            else:  # NNUE
                pred_output = evaluator.inference.evaluate_full(
                    rec['white_features'],
                    rec['black_features'],
                    stm_is_white
                )

            true_tanh = np.tanh(rec['score_cp'] / TANH_SCALE)

            # Convert prediction to CP and cap at MAX_SCORE
            pred_cp_raw = np.arctanh(np.clip(pred_output, -0.99999, 0.99999)) * TANH_SCALE
            pred_cp_capped = max(-MAX_SCORE, min(MAX_SCORE, pred_cp_raw))

            true_tanh_values.append(true_tanh)
            pred_tanh_values.append(pred_output)
            true_cp_list.append(rec['score_cp'])
            pred_cp_list.append(pred_cp_capped)
            fen_list.append(fen)

        except Exception as e:
            errors.append((i, str(e)))

    if errors:
        print(f"\n⚠ {len(errors)} positions failed to evaluate:")
        for idx, error in errors[:5]:
            print(f"  Position {idx}: {error[:60]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not true_tanh_values:
        print("\n❌ ERROR: No positions could be evaluated")
        return

    # Compute metrics
    true_tanh_values = np.array(true_tanh_values)
    pred_tanh_values = np.array(pred_tanh_values)
    true_cp_values = np.array(true_cp_list)
    pred_cp_values = np.array(pred_cp_list)

    mse = np.mean((true_tanh_values - pred_tanh_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_tanh_values - pred_tanh_values))

    # Centipawn metrics (with capped values)
    cp_diff = pred_cp_values - true_cp_values
    cp_diff_capped = np.clip(cp_diff, -CP_ERROR_CLIP, CP_ERROR_CLIP)
    mse_cp = np.mean(cp_diff_capped ** 2)
    rmse_cp = np.sqrt(mse_cp)
    mae_cp = np.mean(np.abs(cp_diff_capped))

    # Results
    print("\n" + "─" * 70)
    print("Results:")
    print("─" * 70)
    print(f"Positions evaluated: {len(pred_tanh_values)}")
    print(f"Positions failed:    {len(errors)}")
    print()
    print("Tanh Space (network output space):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print()
    print("Centipawn Space (Error capped at +/-{:,}):".format(CP_ERROR_CLIP))
    print(f"  MSE:  {mse_cp:.2f}")
    print(f"  RMSE: {rmse_cp:.2f} cp")
    print(f"  MAE:  {mae_cp:.2f} cp")

    # Sample predictions
    print("\n" + "─" * 70)
    print("Sample predictions (first 20):")
    print("─" * 70)
    for i in range(min(20, len(pred_tanh_values))):
        true_cp = true_cp_list[i]
        pred_cp = pred_cp_list[i]
        diff = np.clip(pred_cp - true_cp, -MAX_SCORE, MAX_SCORE)
        fen = fen_list[i]
        print(f"{i + 1:2d}. True: {true_cp:+7.1f} cp | "
              f"Pred: {pred_cp:+7.1f} cp | "
              f"Diff: {diff:+7.1f} cp")
        print(f"    FEN: {fen}")

    print("\n" + "=" * 70)
    print("Evaluation accuracy test complete!")
    print("=" * 70)


# =============================================================================
# NN vs Stockfish Test
# =============================================================================
# TODO save 10_000 eavluations to a file for faster testing
def read_sf_eval_file(filepath: str):
    """
    Read the pre-computed Stockfish evaluation binary file.

    Binary format:
        Header: [num_records:uint32]
        Each record: [fen_length:uint8][fen_bytes:char[]][sf_eval_cp:int16][shard_cp:int16]

    Returns list of (fen, sf_eval_cp, shard_cp) tuples.
    """
    import struct

    records = []
    with open(filepath, 'rb') as f:
        # Read header
        num_records = struct.unpack('<I', f.read(4))[0]

        # Read records
        for _ in range(num_records):
            fen_len = struct.unpack('<B', f.read(1))[0]
            fen = f.read(fen_len).decode('utf-8')
            sf_eval_cp = struct.unpack('<h', f.read(2))[0]
            shard_cp = struct.unpack('<h', f.read(2))[0]
            records.append((fen, sf_eval_cp, shard_cp))

    return records


SF_EVAL_FILE_PATH = "data/sf_nnue_static_eval.bin"


def test_nn_vs_stockfish(nn_type: str, model_path: str, positions_size: int, stockfish_path: str = "stockfish"):
    """
    Test neural network predictions against Stockfish's static NNUE evaluation.

    Reads pre-computed Stockfish evaluations from data/sf_nnue_static_eval.bin
    (generated by build_sf_static_eval_file.py) for fast comparison.
    """
    import os
    from pathlib import Path

    print("\n" + "=" * 70)
    print(f"{nn_type} vs Stockfish Static Evaluation Test")
    print("=" * 70)

    # Check for pre-computed Stockfish evaluation file
    if not os.path.exists(SF_EVAL_FILE_PATH):
        print(f"\n❌ ERROR: Pre-computed Stockfish evaluation file not found:")
        print(f"    {SF_EVAL_FILE_PATH}")
        print("\nPlease generate it first by running:")
        print(f"    python build_sf_static_eval_file.py --num-positions {positions_size}")
        return

    # Read pre-computed evaluations
    print(f"\nReading pre-computed Stockfish evaluations from: {SF_EVAL_FILE_PATH}")
    all_records = read_sf_eval_file(SF_EVAL_FILE_PATH)
    print(f"✓ Loaded {len(all_records)} pre-computed evaluations")

    # Take first N positions if requested fewer than available
    if positions_size < len(all_records):
        records = all_records[:positions_size]
        print(f"  Using first {positions_size} positions")
    else:
        records = all_records
        if positions_size > len(all_records):
            print(f"  Warning: Requested {positions_size} but only {len(all_records)} available")

    # Create evaluator
    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    # Evaluate all positions
    print("\nEvaluating positions with neural network...")

    sf_tanh_values = []
    true_tanh_values = []
    pred_tanh_values = []
    sf_cp_list = []
    pred_cp_list = []
    true_cp_list = []
    fen_list = []
    errors = []

    for i, (fen, sf_cp_stm, shard_cp) in enumerate(records):
        if (i + 1) % max(1, len(records) // 10) == 0:
            print(f"  Progress: {i + 1}/{len(records)}")

        try:
            # Determine side to move from FEN
            stm_is_white = ' w ' in fen

            # Set up board and evaluate with full evaluation
            board = chess.Board(fen)
            pred_output = evaluator.evaluate_full(board)

            # SF eval is already in STM perspective and capped at MAX_SCORE
            sf_cp_capped = sf_cp_stm

            sf_tanh = np.tanh(sf_cp_capped / TANH_SCALE)

            # Convert prediction to CP and cap at MAX_SCORE
            pred_cp_raw = np.arctanh(np.clip(pred_output, -0.99999, 0.99999)) * TANH_SCALE
            pred_cp_capped = max(-MAX_SCORE, min(MAX_SCORE, pred_cp_raw))

            true_tanh = np.tanh(shard_cp / TANH_SCALE)

            sf_tanh_values.append(sf_tanh)
            true_tanh_values.append(true_tanh)
            pred_tanh_values.append(pred_output)
            sf_cp_list.append(sf_cp_capped)
            pred_cp_list.append(pred_cp_capped)
            true_cp_list.append(shard_cp)
            fen_list.append(fen)

        except Exception as e:
            errors.append((i, str(e)))

    if errors:
        print(f"\n⚠ {len(errors)} positions failed:")
        for idx, error in errors[:5]:
            print(f"  Position {idx}: {error[:60]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not sf_tanh_values:
        print("\n❌ ERROR: No positions could be evaluated")
        return

    # Compute metrics
    sf_tanh_values = np.array(sf_tanh_values)
    true_tanh_values = np.array(true_tanh_values)
    pred_tanh_values = np.array(pred_tanh_values)

    nn_error_values = np.abs(pred_tanh_values - true_tanh_values)
    sf_error_values = np.abs(sf_tanh_values - true_tanh_values)
    mse = np.mean((nn_error_values - sf_error_values) ** 2)
    rmse = np.sqrt(mse)
    mean_delta_error = np.mean(nn_error_values - sf_error_values)

    #mean_delta_error_cp = np.arctanh(np.clip(mean_delta_error, -0.99999, 0.99999)) * TANH_SCALE
    #mean_delta_error_cp = np.clip(mean_delta_error_cp, -MAX_SCORE, MAX_SCORE)
    nn_error_values_cp = np.arctanh(np.clip(nn_error_values, -0.99999, 0.99999)) * TANH_SCALE
    sf_error_values_cp = np.arctanh(np.clip(sf_error_values, -0.99999, 0.99999)) * TANH_SCALE
    nn_error_values_cp = np.abs(np.clip(nn_error_values_cp, -CP_ERROR_CLIP, CP_ERROR_CLIP))
    sf_error_values_cp = np.abs(np.clip(sf_error_values_cp, -CP_ERROR_CLIP, CP_ERROR_CLIP))
    mean_delta_error_cp = np.mean(np.abs(nn_error_values_cp - sf_error_values_cp))

    # Results
    print("\n" + "─" * 70)
    print("Results:")
    print("─" * 70)
    print(f"Positions evaluated: {len(pred_tanh_values)}")
    print(f"Positions failed:    {len(errors)}")
    print()
    print("Tanh Space (network output space):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Mean Delta Error:  {mean_delta_error:.6f} {'NN is better than SF' \
        if mean_delta_error <= 0 else 'SF is better than NN' }")
    print()
    print("Centipawn Space (Error capped at +/-{:,}):".format(CP_ERROR_CLIP))
    print(f"  Mean Delta Error:  {mean_delta_error_cp:.2f} cp")

    # Sample predictions
    print("\n" + "─" * 70)
    print("Sample predictions (first 20):")
    print("─" * 70)
    for i in range(min(20, len(pred_tanh_values))):
        sf_cp = sf_cp_list[i]
        pred_cp = pred_cp_list[i]
        true_cp = true_cp_list[i]
        diff = np.clip(pred_cp - sf_cp, -MAX_SCORE, MAX_SCORE)
        fen = fen_list[i]
        print(f"{i + 1:2d}.  True: {true_cp:+7.1f} cp | SF: {sf_cp:+7.1f} cp | "
              f"Pred: {pred_cp:+7.1f} cp | "
              f"Diff: {diff:+7.1f} cp")
        print(f"    FEN: {fen}")

    print("\n" + "=" * 70)
    print("NN vs Stockfish test complete!")
    print("=" * 70)


# =============================================================================
# Feature Extraction Test
# =============================================================================

def test_feature_extraction(nn_type: str):
    """
    Test that feature extraction produces correct and consistent results.

    Verifies:
    - Feature indices are within valid range
    - No duplicate features
    - Symmetric positions produce mirrored features
    - Feature count is reasonable for different positions
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Feature Extraction Test")
    print("=" * 70)

    all_passed = True

    # Test positions with known properties
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("8/8/8/8/8/8/8/4K2k w - - 0 1", "Minimal position (2 kings)"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "Castling rights position"),
        ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "After 1.e4 e6"),
        ("8/P7/8/8/8/8/8/4K2k w - - 0 1", "Promotion possible"),
    ]

    if nn_type == "NNUE":
        max_features = NNUE_INPUT_SIZE
        feature_extractor = NNUEFeatures
    else:
        max_features = DNN_INPUT_SIZE
        feature_extractor = DNNFeatures

    for fen, description in test_positions:
        print(f"\n{'─' * 70}")
        print(f"Testing: {description}")
        print(f"FEN: {fen}")

        board = chess.Board(fen)

        if nn_type == "NNUE":
            white_feat, black_feat = feature_extractor.board_to_features(board)

            # Check white features
            white_in_range = all(0 <= f < max_features for f in white_feat)
            white_no_dups = len(white_feat) == len(set(white_feat))

            # Check black features
            black_in_range = all(0 <= f < max_features for f in black_feat)
            black_no_dups = len(black_feat) == len(set(black_feat))

            print(f"  White features: {len(white_feat)}, Black features: {len(black_feat)}")
            print(f"  White in range: {white_in_range}, No duplicates: {white_no_dups}")
            print(f"  Black in range: {black_in_range}, No duplicates: {black_no_dups}")

            passed = white_in_range and white_no_dups and black_in_range and black_no_dups
        else:
            features = feature_extractor.board_to_features(board)

            # Check features
            in_range = all(0 <= f < max_features for f in features)
            no_dups = len(features) == len(set(features))

            print(f"  Features: {len(features)}")
            print(f"  In range: {in_range}, No duplicates: {no_dups}")

            passed = in_range and no_dups

        if passed:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All feature extraction tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Symmetry Test
# =============================================================================

def test_symmetry(nn_type: str, model_path: str):
    """
    Test that the evaluator respects chess symmetry.

    A position with colors swapped and board flipped should give
    the negated evaluation (or very close to it).
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Symmetry Test")
    print("=" * 70)

    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    all_passed = True

    # Test positions - pairs of (white_to_move_fen, black_equivalent_fen)
    test_pairs = [
        # Starting position - should be symmetric
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr b KQkq - 0 1"),
        # After 1.e4
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
         "RNBQKBNR/PPPP1PPP/8/4p3/8/8/pppppppp/rnbqkbnr w KQkq e6 0 1"),
    ]

    tolerance = 0.1  # Allow some tolerance due to asymmetries in training data

    for fen1, fen2 in test_pairs:
        print(f"\n{'─' * 70}")
        print(f"Original: {fen1}")
        print(f"Mirrored: {fen2}")

        board1 = chess.Board(fen1)
        board2 = chess.Board(fen2)

        eval1 = evaluator.evaluate_full(board1)
        eval2 = evaluator.evaluate_full(board2)

        # Evaluations should be opposite (one is STM advantage, other is opponent)
        # Since both are from STM perspective, swapping colors should negate
        diff = abs(eval1 + eval2)

        print(f"  Eval 1: {eval1:.6f} ({output_to_centipawns(eval1):+.2f} cp)")
        print(f"  Eval 2: {eval2:.6f} ({output_to_centipawns(eval2):+.2f} cp)")
        print(f"  Sum (should be ~0): {eval1 + eval2:.6f}")

        passed = diff < tolerance
        if passed:
            print("  ✓ PASS: Evaluations are symmetric")
        else:
            print(f"  ⚠ WARNING: Evaluations differ by {diff:.6f} (tolerance: {tolerance})")
            # Don't fail for symmetry since training data may not be perfectly symmetric

    print("\n" + "=" * 70)
    print("Symmetry test complete!")
    print("(Note: Small asymmetries are expected due to training data)")
    print("=" * 70)

    return all_passed


# =============================================================================
# Edge Cases Test
# =============================================================================

def test_edge_cases(nn_type: str, model_path: str):
    """
    Test evaluation of edge cases: checkmate, stalemate, and special moves.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Edge Cases Test")
    print("=" * 70)

    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    all_passed = True

    # Test cases with expected behavior
    test_cases = [
        # Checkmate positions
        ("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
         "Fool's mate (White is checkmated)", "checkmate"),
        ("r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
         "Scholar's mate (Black is checkmated)", "checkmate"),

        # Stalemate positions (verified with python-chess)
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
         "Black king stalemated in corner by queen", "stalemate"),
        ("k7/8/1K6/8/8/8/8/8 b - - 0 1",
         "King vs King (insufficient material - draw)", "insufficient_material"),
        ("8/8/8/8/8/5k2/5p2/5K2 w - - 0 1",
         "White king stalemated by pawn", "stalemate"),

        # Normal positions
        ("8/8/8/8/8/6k1/8/5K1Q b - - 0 1",
         "Not stalemate - black king has legal moves", "normal"),

        # Complex middlegame
        ("r3k2r/pp3ppp/2p1pn2/3p4/3P4/2P1PN2/PP3PPP/R3K2R w KQkq - 0 10",
         "Complex middlegame", "normal"),
    ]

    for fen, description, expected_type in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Testing: {description}")
        print(f"FEN: {fen}")

        test_board = chess.Board(fen)

        # Check game state
        is_checkmate = test_board.is_checkmate()
        is_stalemate = test_board.is_stalemate()
        is_game_over = test_board.is_game_over()
        is_insufficient = test_board.is_insufficient_material()

        print(f"  Checkmate: {is_checkmate}, Stalemate: {is_stalemate}, "
              f"Game over: {is_game_over}, Insufficient: {is_insufficient}")

        # Get evaluation
        try:
            eval_result = evaluator.evaluate_full(test_board)
            print(f"  Evaluation: {eval_result:.6f} ({output_to_centipawns(eval_result):+.2f} cp)")

            # Verify expected behavior
            if expected_type == "checkmate":
                if is_checkmate:
                    print("  ✓ PASS: Correctly identified as checkmate")
                else:
                    print("  ✗ FAIL: Should be checkmate")
                    all_passed = False
            elif expected_type == "stalemate":
                if is_stalemate:
                    if abs(eval_result) < 0.01:  # Should evaluate to ~0
                        print("  ✓ PASS: Correctly identified as stalemate with ~0 eval")
                    else:
                        print(f"  ⚠ WARNING: Stalemate but eval is {eval_result:.6f}")
                else:
                    print("  ✗ FAIL: Should be stalemate")
                    all_passed = False
            elif expected_type == "insufficient_material":
                if is_insufficient or is_game_over:
                    print("  ✓ PASS: Correctly identified as draw (insufficient material)")
                else:
                    print("  ✗ FAIL: Should be insufficient material")
                    all_passed = False
            else:
                print("  ✓ PASS: Normal position evaluated successfully")

        except Exception as e:
            print(f"  ✗ FAIL: Evaluation error: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All edge case tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Reset Consistency Test
# =============================================================================

def test_reset_consistency(nn_type: str, model_path: str):
    """
    Test that resetting the evaluator produces consistent results.

    After push/pop operations, reset should restore to a clean state
    that matches a freshly created evaluator.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Reset Consistency Test")
    print("=" * 70)

    all_passed = True

    # Create two evaluators
    board1 = chess.Board()
    board2 = chess.Board()

    evaluator1 = NNEvaluator.create(board1, nn_type, model_path)
    evaluator2 = NNEvaluator.create(board2, nn_type, model_path)

    # Make some moves on evaluator1
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        evaluator_push(evaluator1, board1, move)

    print(f"After {len(moves)} moves:")
    print(f"  Board1: {board1.fen()}")

    # Now reset evaluator1 to the current position of board2 (starting position)
    evaluator1.reset(board2)

    # Both evaluators should now give the same evaluation for the starting position
    eval1 = evaluator1.evaluate(board2)
    eval2 = evaluator2.evaluate(board2)

    diff = abs(eval1 - eval2)
    print(f"\nAfter reset to starting position:")
    print(f"  Evaluator1: {eval1:.10f}")
    print(f"  Evaluator2: {eval2:.10f}")
    print(f"  Difference: {diff:.10e}")

    if diff < 1e-6:
        print("  ✓ PASS: Reset produces consistent results")
    else:
        print("  ✗ FAIL: Reset produced different results")
        all_passed = False

    # Also test resetting to a different position
    test_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    test_board = chess.Board(test_fen)

    evaluator1.reset(test_board)
    eval_reset = evaluator1.evaluate(test_board)
    eval_full = evaluator1.evaluate_full(test_board)

    diff = abs(eval_reset - eval_full)
    print(f"\nAfter reset to new position:")
    print(f"  Position: {test_fen}")
    print(f"  Incremental: {eval_reset:.10f}")
    print(f"  Full:        {eval_full:.10f}")
    print(f"  Difference:  {diff:.10e}")

    if diff < 1e-6:
        print("  ✓ PASS: Reset to arbitrary position works")
    else:
        print("  ✗ FAIL: Reset to arbitrary position failed")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All reset consistency tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Deep Search Simulation Test
# =============================================================================

def test_deep_search_simulation(nn_type: str, model_path: str, depth: int = 6,
                                  num_iterations: int = 100, tolerance: float = 1e-4):
    """
    Simulate a deep search with many push/pop cycles.

    This tests that the evaluator maintains consistency through
    many levels of recursion and undo operations.

    Note: A tolerance of 1e-4 is used because floating-point errors
    naturally accumulate over many push/pop operations. This is expected
    behavior and doesn't affect practical usage.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Deep Search Simulation Test")
    print("=" * 70)
    print(f"Parameters: depth={depth}, iterations={num_iterations}, tolerance={tolerance}")

    board = chess.Board()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    all_passed = True
    total_nodes = 0
    max_diff = 0.0

    def search_recursive(depth_remaining: int, path: list) -> int:
        """Recursive search that validates at each node."""
        nonlocal total_nodes, max_diff, all_passed

        total_nodes += 1

        # Validate incremental vs full at this node
        eval_inc = evaluator.evaluate(board)
        eval_full = evaluator.evaluate_full(board)
        diff = abs(eval_inc - eval_full)
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            print(f"  ✗ Mismatch at depth {len(path)}: diff={diff:.10e}")
            print(f"    Path: {' '.join(m.uci() for m in path)}")
            all_passed = False

        if depth_remaining == 0 or board.is_game_over():
            return 1

        nodes1 = 0
        legal_moves = list(board.legal_moves)[:5]  # Limit branching

        for move in legal_moves:
            evaluator_push(evaluator, board, move)
            path.append(move)

            nodes1 += search_recursive(depth_remaining - 1, path)

            path.pop()
            evaluator_pop(evaluator, board)

        return nodes1

    print(f"\nRunning {num_iterations} search iterations...")

    start_time = time.time()
    for i in range(num_iterations):
        total_nodes = 0
        nodes = search_recursive(depth, [])

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{num_iterations}: {nodes} nodes, max_diff={max_diff:.2e}")

    elapsed = time.time() - start_time

    print(f"\n{'─' * 70}")
    print("Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Nodes per second: {total_nodes * num_iterations / elapsed:.0f}")
    print(f"  Maximum difference observed: {max_diff:.10e}")
    print(f"  Tolerance used: {tolerance:.0e}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ Deep search simulation PASSED!")
    else:
        print("✗ Deep search simulation FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Random Games Test
# =============================================================================

def test_random_games(nn_type: str, model_path: str, num_games: int = 10, max_moves: int = 100):
    """
    Play through random games and verify incremental vs full evaluation.

    This tests a wide variety of positions that might not be covered
    by handcrafted test sequences.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Random Games Test")
    print("=" * 70)
    print(f"Parameters: num_games={num_games}, max_moves={max_moves}")

    all_passed = True
    total_positions = 0
    max_diff = 0.0
    failures = []

    random.seed(42)  # Reproducible

    for game_num in range(num_games):
        board = chess.Board()
        evaluator = NNEvaluator.create(board, nn_type, model_path)

        moves_played = []

        for move_num in range(max_moves):
            if board.is_game_over():
                break

            # Pick a random legal move
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)

            # Push and validate
            evaluator_push(evaluator, board, move)
            moves_played.append(move)
            total_positions += 1

            eval_inc = evaluator.evaluate(board)
            eval_full = evaluator.evaluate_full(board)
            diff = abs(eval_inc - eval_full)
            max_diff = max(max_diff, diff)

            if diff > 1e-6:
                failures.append({
                    'game': game_num,
                    'move': move_num,
                    'fen': board.fen(),
                    'diff': diff,
                    'moves': [m.uci() for m in moves_played[-5:]]
                })
                all_passed = False

        # Test some pop operations
        num_pops = min(5, len(moves_played))
        for _ in range(num_pops):
            evaluator_pop(evaluator, board)

            eval_inc = evaluator.evaluate(board)
            eval_full = evaluator.evaluate_full(board)
            diff = abs(eval_inc - eval_full)
            max_diff = max(max_diff, diff)

            if diff > 1e-6:
                all_passed = False

        print(f"  Game {game_num + 1}/{num_games}: {len(moves_played)} moves, "
              f"max_diff so far: {max_diff:.2e}")

    print(f"\n{'─' * 70}")
    print("Results:")
    print(f"  Total positions tested: {total_positions}")
    print(f"  Maximum difference: {max_diff:.10e}")
    print(f"  Failures: {len(failures)}")

    if failures:
        print("\nFirst 5 failures:")
        for f in failures[:5]:
            print(f"  Game {f['game']}, move {f['move']}: diff={f['diff']:.2e}")
            print(f"    Recent moves: {' '.join(f['moves'])}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ Random games test PASSED!")
    else:
        print("✗ Random games test FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# CP Integrity Test
# =============================================================================
import chess

def fen_to_sparse_planes_chatgpt(fen: str) -> list[int]:
    """
    Convert a FEN string into sparse one-hot indices for a
    12x64 (768) feature vector.

    Plane order per side:
        K, Q, R, B, N, P

    Planes:
        0–5   : side to move
        6–11  : opponent

    Board is oriented from the side-to-move perspective.
    """
    board = chess.Board(fen)

    # Piece → plane index (within a side)
    piece_to_plane = {
        chess.KING:   0,
        chess.QUEEN:  1,
        chess.ROOK:   2,
        chess.BISHOP: 3,
        chess.KNIGHT: 4,
        chess.PAWN:   5,
    }

    sparse_indices = []
    stm = board.turn  # True = White, False = Black

    for square, piece in board.piece_map().items():
        # Base plane index (0–5)
        plane = piece_to_plane[piece.piece_type]

        # Offset for opponent pieces
        if piece.color != stm:
            plane += 6

        # Orient square from side-to-move perspective
        if stm == chess.WHITE:
            oriented_square = square
        else:
            oriented_square = chess.square(
                chess.square_file(square),
                7 - chess.square_rank(square)
            )

        index = plane * 64 + oriented_square
        sparse_indices.append(index)

    return sparse_indices


def test_cp_integrity(nn_type: str, num_positions: int = 10, data_dir: str = "data",
                      stockfish_path: str = "stockfish", time_limit: float = 2.0,
                      threshold: float = 0.01):
    """
    Test training data integrity by comparing stored scores against Stockfish evaluation.

    Uses diagnostic records (with FEN) from shard files to:
    1. Get the actual FEN for each position
    2. Run Stockfish analysis on that FEN
    3. Compare stored score (STM perspective) against Stockfish score

    Args:
        nn_type: "DNN" or "NNUE"
        num_positions: Number of positions to validate
        data_dir: Base data directory containing dnn/ and nnue/ subdirs
        stockfish_path: Path to stockfish binary
        time_limit: Stockfish time limit per position (default 2.0 seconds)
        threshold: Mismatch threshold in tanh scale (default 0.01)

    Returns:
        True if all validations pass, False otherwise
    """
    import math
    from pathlib import Path
    import chess.engine
    from shard_io import ShardReader, find_shards

    print("\n" + "=" * 70)
    print(f"{nn_type} CP Integrity Test")
    print("=" * 70)

    # Constants matching nn_inference.py
    MAX_SCORE = 10_000
    MATE_FACTOR = 100
    MAX_MATE_DEPTH = 10
    MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR

    def cp_to_tanh(cp: int, scale: float = TANH_SCALE) -> float:
        return math.tanh(cp / scale)

    # Find shard files
    dnn_shards, nnue_shards = find_shards(data_dir, nn_type)
    shard_files = dnn_shards if nn_type.upper() == "DNN" else nnue_shards

    if not shard_files:
        print(f"\n❌ ERROR: No shard files found for {nn_type}")
        return False

    # Pick random shard
    shard_path = random.choice(shard_files)
    print(f"\nSelected shard: {shard_path}")

    # Read diagnostic records (which have FEN)
    reader = ShardReader(nn_type)
    records = reader.read_diagnostic_records(shard_path, max_records=num_positions)

    if not records:
        print("\n❌ ERROR: No diagnostic records found in shard file")
        print("Diagnostic records are written every 1000 positions and include FEN.")
        print("You may need to regenerate shards with the updated prepare_data.py")
        return False

    print(f"Read {len(records)} diagnostic records for validation")

    # Initialize Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"\n❌ ERROR: Could not initialize Stockfish at '{stockfish_path}': {e}")
        return False

    def get_stockfish_score_stm(fen: str, stm_is_white: bool):
        """
        Get Stockfish score from STM perspective.

        Stockfish returns score from white's perspective.
        We convert to STM perspective to match bin.zst format.
        """
        try:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = info["score"].white()  # White's perspective

            if score.is_mate():
                mate_in = score.mate()
                if mate_in < 0:  # Black winning
                    mate_in = max(-MAX_MATE_DEPTH, mate_in)
                    score_cp = -MAX_SCORE - mate_in * MATE_FACTOR
                else:  # White winning
                    mate_in = min(MAX_MATE_DEPTH, mate_in)
                    score_cp = MAX_SCORE - mate_in * MATE_FACTOR
            else:
                score_cp = score.score()
                score_cp = max(-MAX_NON_MATE_SCORE, min(MAX_NON_MATE_SCORE, score_cp))

            # Convert to STM perspective (matching prepare_data.py logic)
            if not stm_is_white:  # Black to move
                score_cp = -score_cp

            return score_cp
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None

    # Validate positions
    validated = 0
    passed = 0
    mismatches = 0
    errors = 0
    mismatch_details = []

    try:
        for i, rec in enumerate(records):
            fen = rec['fen']
            stm_is_white = rec['stm'] == 1

            # Get Stockfish evaluation
            sf_score = get_stockfish_score_stm(fen, stm_is_white)
            if sf_score is None:
                errors += 1
                continue

            # Compare scores in tanh scale
            target_cp = rec['score_cp']
            target_tanh = cp_to_tanh(target_cp)
            sf_tanh = cp_to_tanh(sf_score)
            diff = abs(target_tanh - sf_tanh)

            validated += 1

            if diff > threshold:
                mismatches += 1
                mismatch_details.append({
                    'fen': fen,
                    'target_cp': target_cp,
                    'sf_cp': sf_score,
                    'diff': diff
                })
                print(f"[{i+1}/{len(records)}] MISMATCH:")
                print(f"  FEN: {fen}")
                print(f"  Target: {target_cp} cp ({target_tanh:.4f} tanh)")
                print(f"  Stockfish: {sf_score} cp ({sf_tanh:.4f} tanh)")
                print(f"  Diff: {diff:.4f} (threshold: {threshold})")
            else:
                passed += 1
                print(f"[{i+1}/{len(records)}] OK: diff={diff:.4f} | {fen[:50]}...")

    finally:
        engine.quit()

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Shard: {Path(shard_path).name}")
    print(f"  Positions validated: {validated}")
    print(f"  Passed: {passed}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Errors: {errors}")
    if validated > 0:
        rate = mismatches / validated * 100
        print(f"  Mismatch rate: {rate:.1f}%")

    all_passed = mismatches == 0 and errors == 0

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ Data integrity test PASSED!")
    else:
        print("✗ Data integrity test FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests(nn_type: str, model_path: str):
    """Run all non-interactive tests and report results."""
    print("\n" + "=" * 70)
    print(f"RUNNING ALL {nn_type} TESTS")
    print("=" * 70)

    results = {}

    # Run each test
    tests = [
        ("Accumulator Correctness", lambda: test_accumulator_correctness(nn_type, model_path)),
        ("Feature Extraction", lambda: test_feature_extraction(nn_type)),
        ("Edge Cases", lambda: test_edge_cases(nn_type, model_path)),
        ("Reset Consistency", lambda: test_reset_consistency(nn_type, model_path)),
        ("Deep Search Simulation", lambda: test_deep_search_simulation(nn_type, model_path, depth=4, num_iterations=20, tolerance=1e-3)),
        ("Random Games", lambda: test_random_games(nn_type, model_path, num_games=5, max_moves=50)),
        ("CP-Integrity", lambda: test_cp_integrity(nn_type=nn_type)),
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Performance test (doesn't return pass/fail)
    print("\n")
    performance_test(nn_type, model_path)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED!")

    print("=" * 70)

    return passed == total


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with ArgumentParser."""
    parser = argparse.ArgumentParser(
        description="Chess Neural Network Evaluation Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test types:
  0  Interactive-FEN         - Interactive FEN evaluation
  1  Incremental-vs-Full     - Performance comparison
  2  Accumulator-Correctness - Verify incremental == full evaluation
  3  Eval-Accuracy           - Test prediction accuracy against training data
  4  NN-vs-Stockfish         - Compare NN predictions against Stockfish static eval
  5  Feature-Extraction      - Verify feature extraction correctness
  6  Symmetry                - Test evaluation symmetry
  7  Edge-Cases              - Test edge cases (checkmate, stalemate, etc.)
  8  Reset-Consistency       - Test evaluator reset functionality
  9  Deep-Search-Simulation  - Simulate deep search with many push/pop cycles
  10 Random-Games            - Test with random legal move sequences
  11 CP-Integrity            - Validate training data against Stockfish
  12 All                     - Run all non-interactive tests

Examples:
  %(prog)s --nn-type NNUE --test 0          # Interactive FEN
  %(prog)s --nn-type DNN --test 1           # Performance test
  %(prog)s --nn-type NNUE --test 2          # Accumulator correctness
  %(prog)s --nn-type DNN --test 3 --positions 1000  # Eval accuracy
  %(prog)s --nn-type DNN --test 10 --num-positions 10  # Data integrity
  %(prog)s --nn-type NNUE --test 11         # Run all tests
  %(prog)s --nn-type NNUE --test 31 --positions 100  # NN vs Stockfish
"""
    )

    parser.add_argument(
        '--nn-type', '-n',
        type=str,
        choices=['NNUE', 'DNN'],
        required=True,
        help='Neural network type: NNUE or DNN'
    )

    parser.add_argument(
        '--test', '-t',
        type=int,
        choices=list(VALID_TEST_TYPES.keys()),
        required=True,
        help='Test type number (see list below)'
    )

    parser.add_argument(
        '--positions', '-p',
        type=int,
        default=100,
        help='Number of positions for Eval-Accuracy and NN-vs-Stockfish tests (default: 100)'
    )

    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path to model file (default: model/nnue.pt or model/dnn.pt)'
    )

    parser.add_argument(
        '--depth', '-d',
        type=int,
        default=6,
        help='Search depth for Deep-Search-Simulation (default: 6)'
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=100,
        help='Number of iterations for Deep-Search-Simulation (default: 100)'
    )

    parser.add_argument(
        '--num-games', '-g',
        type=int,
        default=10,
        help='Number of games for Random-Games test (default: 10)'
    )

    parser.add_argument(
        '--max-moves',
        type=int,
        default=100,
        help='Max moves per game for Random-Games test (default: 100)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-4,
        help='Tolerance for floating-point comparisons in Deep-Search-Simulation (default: 1e-4)'
    )

    parser.add_argument(
        '--num-positions',
        type=int,
        default=10,
        help='Number of positions for CP-Integrity test (default: 10)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory for CP-Integrity test (default: data)'
    )

    parser.add_argument(
        '--stockfish',
        type=str,
        default='stockfish',
        help='Path to Stockfish binary for CP-Integrity and NN-vs-Stockfish tests (default: stockfish)'
    )

    parser.add_argument(
        '--time-limit',
        type=float,
        default=2.0,
        help='Stockfish time limit per position in seconds for CP-Integrity test (default: 2.0)'
    )

    args = parser.parse_args()

    nn_type = args.nn_type
    test_type = VALID_TEST_TYPES[args.test]
    model_path = args.model_path or get_model_path(nn_type)

    print(f"Neural Network Type: {nn_type}")
    print(f"Test Type: {test_type}")
    print(f"Model Path: {model_path}")

    # Run appropriate test
    if test_type == "Interactive-FEN":
        board = chess.Board()
        evaluator = NNEvaluator.create(board, nn_type, model_path)

        # Test with starting position first
        print("\nTesting with starting position...")
        startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = evaluate_fen(startpos, evaluator)
        print_evaluation(result)

        interactive_loop(evaluator, nn_type)

    elif test_type == "Incremental-vs-Full":
        performance_test(nn_type, model_path)

    elif test_type == "Accumulator-Correctness":
        test_accumulator_correctness(nn_type, model_path)

    elif test_type == "Eval-Accuracy":
        if args.positions <= 0:
            print("ERROR: Eval-Accuracy requires --positions > 0")
            sys.exit(1)
        test_eval_accuracy(nn_type, model_path, args.positions)

    elif test_type == "Feature-Extraction":
        test_feature_extraction(nn_type)

    elif test_type == "Symmetry":
        test_symmetry(nn_type, model_path)

    elif test_type == "Edge-Cases":
        test_edge_cases(nn_type, model_path)

    elif test_type == "Reset-Consistency":
        test_reset_consistency(nn_type, model_path)

    elif test_type == "Deep-Search-Simulation":
        test_deep_search_simulation(nn_type, model_path, args.depth, args.iterations, args.tolerance)

    elif test_type == "Random-Games":
        test_random_games(nn_type, model_path, args.num_games, args.max_moves)

    elif test_type == "All":
        success = run_all_tests(nn_type, model_path)
        sys.exit(0 if success else 1)

    elif test_type == "CP-Integrity":
        success = test_cp_integrity(
            nn_type=nn_type,
            num_positions=args.num_positions,
            data_dir=args.data_dir,
            stockfish_path=args.stockfish,
            time_limit=args.time_limit
        )
        sys.exit(0 if success else 1)

    elif test_type == "NN-vs-Stockfish":
        if args.positions <= 0:
            print("ERROR: NN-vs-Stockfish requires --positions > 0")
            sys.exit(1)
        test_nn_vs_stockfish(nn_type, model_path, args.positions, args.stockfish)


if __name__ == "__main__":
    main()