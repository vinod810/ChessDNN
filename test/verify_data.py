#!/usr/bin/env python3
"""
verify_data.py - Verify data correctness in DNN/NNUE shards.

Uses diagnostic records (marker=0xFF) to compare stored features against
features recomputed from the embedded FEN string.

Features:
- Tests both DNN and NNUE shard formats
- Stops immediately on first mismatch
- Only prints details when there is a mismatch
"""

import sys
import os
import argparse
import numpy as np

# Check what TANH_SCALE is set to
try:
    from nn_inference import TANH_SCALE, MAX_SCORE

    print(f"TANH_SCALE = {TANH_SCALE}")
    print(f"MAX_SCORE = {MAX_SCORE}")
except ImportError:
    print("Could not import from nn_inference, using defaults")
    TANH_SCALE = 400
    MAX_SCORE = 10000

import chess
from shard_io import ShardReader, find_shards


def dnn_fen_to_sparse_planes(fen: str) -> list[int]:
    import chess

    def fen_to_sparse_planes(fen: str) -> list[int]:
        """
        Sparse one-hot indices for a 12x64 = 768 feature vector.

        Plane order (python-chess piece order):
            [P, N, B, R, Q, K]  -> side to move
            [p, n, b, r, q, k] -> opponent
        """
        board = chess.Board(fen)

        # python-chess piece_type values:
        # PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
        # Map directly to 0–5 by subtracting 1
        piece_to_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        sparse_indices = []
        stm = board.turn  # True = White, False = Black

        for square, piece in board.piece_map().items():
            # Base plane from python-chess piece order
            plane = piece_to_plane[piece.piece_type]

            # Opponent offset
            if piece.color != stm:
                plane += 6

            # Side-to-move perspective
            if stm == chess.WHITE:
                oriented_square = square
            else:
                oriented_square = chess.square(
                    chess.square_file(square),
                    7 - chess.square_rank(square)
                )

            sparse_indices.append(plane * 64 + oriented_square)

        return sorted(sparse_indices)

    # """
    # Compute DNN sparse features from FEN.
    #
    # Plane order (python-chess piece order):
    #     [P, N, B, R, Q, K]  -> side to move (planes 0-5)
    #     [p, n, b, r, q, k]  -> opponent (planes 6-11)
    #
    # Feature index = plane * 64 + oriented_square
    # """
    # board = chess.Board(fen)
    #
    # sparse_indices = []
    # stm = board.turn
    #
    # for square, piece in board.piece_map().items():
    #     # plane = piece_type - 1 (PAWN=0, KNIGHT=1, ..., KING=5)
    #     plane = piece.piece_type - 1
    #
    #     # Opponent offset
    #     if piece.color != stm:
    #         plane += 6
    #
    #     # Side-to-move perspective: flip board for black
    #     if stm == chess.WHITE:
    #         oriented_square = square
    #     else:
    #         oriented_square = chess.square(
    #             chess.square_file(square),
    #             7 - chess.square_rank(square)
    #         )
    #
    #     sparse_indices.append(plane * 64 + oriented_square)
    #
    # return sorted(sparse_indices)


def nnue_fen_to_halfkp_sparse(fen: str):
    import chess

    def fen_to_halfkp_sparse(fen: str):
        """
        Encode HalfKP features in sparse format.

        Returns:
            white_features: list[int]
            black_features: list[int]

        Feature index range:
            0 .. (64 * 64 * 5 * 2) - 1
        """
        board = chess.Board(fen)

        # Non-king piece mapping (python-chess order)
        piece_to_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
        }

        white_features = []
        black_features = []

        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue

            piece_type = piece_to_index[piece.piece_type]

            # ---------- WHITE perspective ----------
            wk = white_king
            psq = square
            ksq = wk

            color = 0 if piece.color == chess.WHITE else 1

            index = (
                    (((ksq * 64 + psq) * 5 + piece_type) * 2)
                    + color
            )
            white_features.append(index)

            # ---------- BLACK perspective ----------
            bk = black_king

            ksq = chess.square(
                chess.square_file(bk),
                7 - chess.square_rank(bk)
            )
            psq = chess.square(
                chess.square_file(square),
                7 - chess.square_rank(square)
            )

            color = 0 if piece.color == chess.BLACK else 1

            index = (
                    (((ksq * 64 + psq) * 5 + piece_type) * 2)
                    + color
            )
            black_features.append(index)

        return sorted(white_features), sorted(black_features)

    # """
    # Compute NNUE HalfKP sparse features from FEN.
    #
    # Returns:
    #     white_features: list[int]
    #     black_features: list[int]
    #
    # Feature index formula: king_sq * 640 + piece_sq * 10 + (type_idx + color_idx * 5)
    # """
    # board = chess.Board(fen)
    #
    # white_features = []
    # black_features = []
    #
    # white_king = board.king(chess.WHITE)
    # black_king = board.king(chess.BLACK)
    #
    # for square, piece in board.piece_map().items():
    #     if piece.piece_type == chess.KING:
    #         continue
    #
    #     # type_idx = piece_type - 1 (PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4)
    #     piece_type = piece.piece_type - 1
    #
    #     # ---------- WHITE perspective ----------
    #     ksq = white_king
    #     psq = square
    #     is_friendly = piece.color == chess.WHITE
    #     color_idx = 1 if is_friendly else 0
    #
    #     index = ksq * 640 + psq * 10 + (piece_type + color_idx * 5)
    #     white_features.append(index)
    #
    #     # ---------- BLACK perspective ----------
    #     # Flip both king and piece squares for black's perspective
    #     ksq = chess.square(
    #         chess.square_file(black_king),
    #         7 - chess.square_rank(black_king)
    #     )
    #     psq = chess.square(
    #         chess.square_file(square),
    #         7 - chess.square_rank(square)
    #     )
    #     is_friendly = piece.color == chess.BLACK
    #     color_idx = 1 if is_friendly else 0
    #
    #     index = ksq * 640 + psq * 10 + (piece_type + color_idx * 5)
    #     black_features.append(index)
    #
    # return sorted(white_features), sorted(black_features)


def verify_dnn_shard(shard_path: str, max_records: int = 10) -> bool:
    """
    Verify DNN diagnostic records by comparing stored features against FEN.

    Returns True if all records match, False on first mismatch.
    """
    print(f"\nVerifying DNN shard: {shard_path}")

    reader = ShardReader("DNN")
    records = reader.read_diagnostic_records(shard_path, max_records)

    if not records:
        print("  WARNING: No diagnostic records found in shard")
        return True

    print(f"  Found {len(records)} diagnostic records to verify")

    for rec in records:
        fen = rec['fen']
        stored_features = sorted(rec['features'])
        stored_stm = rec['stm']

        # Recompute features from FEN
        expected_features = dnn_fen_to_sparse_planes(fen)

        # Check STM
        board = chess.Board(fen)
        expected_stm = 1 if board.turn == chess.WHITE else 0

        if stored_stm != expected_stm:
            print(f"\n  STM MISMATCH at position {rec['position_idx']}!")
            print(f"  FEN: {fen}")
            print(f"  Stored STM: {stored_stm}, Expected STM: {expected_stm}")
            return False

        if stored_features != expected_features:
            print(f"\n  FEATURE MISMATCH at position {rec['position_idx']}!")
            print(f"  FEN: {fen}")
            print(f"  Score: {rec['score_cp']} cp")
            print(f"  STM: {'White' if stored_stm == 1 else 'Black'}")
            print(f"  Stored features ({len(stored_features)}):   {stored_features}")
            print(f"  Expected features ({len(expected_features)}): {expected_features}")

            # Show differences
            stored_set = set(stored_features)
            expected_set = set(expected_features)
            only_in_stored = stored_set - expected_set
            only_in_expected = expected_set - stored_set

            if only_in_stored:
                print(f"  Only in stored: {sorted(only_in_stored)}")
            if only_in_expected:
                print(f"  Only in expected: {sorted(only_in_expected)}")

            return False

    print(f"  ✓ All {len(records)} diagnostic records verified successfully")
    return True


def verify_nnue_shard(shard_path: str, max_records: int = 10) -> bool:
    """
    Verify NNUE diagnostic records by comparing stored features against FEN.

    Returns True if all records match, False on first mismatch.
    """
    print(f"\nVerifying NNUE shard: {shard_path}")

    reader = ShardReader("NNUE")
    records = reader.read_diagnostic_records(shard_path, max_records)

    if not records:
        print("  WARNING: No diagnostic records found in shard")
        return True

    print(f"  Found {len(records)} diagnostic records to verify")

    for rec in records:
        fen = rec['fen']
        stored_white = sorted(rec['white_features'])
        stored_black = sorted(rec['black_features'])
        stored_stm = rec['stm']

        # Recompute features from FEN
        expected_white, expected_black = nnue_fen_to_halfkp_sparse(fen)

        # Check STM
        board = chess.Board(fen)
        expected_stm = 1 if board.turn == chess.WHITE else 0

        if stored_stm != expected_stm:
            print(f"\n  STM MISMATCH at position {rec['position_idx']}!")
            print(f"  FEN: {fen}")
            print(f"  Stored STM: {stored_stm}, Expected STM: {expected_stm}")
            return False

        if stored_white != expected_white:
            print(f"\n  WHITE FEATURES MISMATCH at position {rec['position_idx']}!")
            print(f"  FEN: {fen}")
            print(f"  Score: {rec['score_cp']} cp")
            print(f"  STM: {'White' if stored_stm == 1 else 'Black'}")
            print(f"  Stored white ({len(stored_white)}):   {stored_white}")
            print(f"  Expected white ({len(expected_white)}): {expected_white}")

            stored_set = set(stored_white)
            expected_set = set(expected_white)
            only_in_stored = stored_set - expected_set
            only_in_expected = expected_set - stored_set

            if only_in_stored:
                print(f"  Only in stored: {sorted(only_in_stored)}")
            if only_in_expected:
                print(f"  Only in expected: {sorted(only_in_expected)}")

            return False

        if stored_black != expected_black:
            print(f"\n  BLACK FEATURES MISMATCH at position {rec['position_idx']}!")
            print(f"  FEN: {fen}")
            print(f"  Score: {rec['score_cp']} cp")
            print(f"  STM: {'White' if stored_stm == 1 else 'Black'}")
            print(f"  Stored black ({len(stored_black)}):   {stored_black}")
            print(f"  Expected black ({len(expected_black)}): {expected_black}")

            stored_set = set(stored_black)
            expected_set = set(expected_black)
            only_in_stored = stored_set - expected_set
            only_in_expected = expected_set - stored_set

            if only_in_stored:
                print(f"  Only in stored: {sorted(only_in_stored)}")
            if only_in_expected:
                print(f"  Only in expected: {sorted(only_in_expected)}")

            return False

    print(f"  ✓ All {len(records)} diagnostic records verified successfully")
    return True


def analyze_shard(shard_path: str, nn_type: str):
    """Analyze a shard for statistics."""
    print(f"\n=== Shard Analysis: {shard_path} ===")

    reader = ShardReader(nn_type)
    positions = reader.read_all_positions(shard_path, include_fen=False)

    if not positions:
        print("  No positions found")
        return

    scores = np.array([p['score_cp'] for p in positions])
    diagnostic_count = sum(1 for p in positions if p.get('is_diagnostic', False))

    # Compute tanh targets
    targets = np.tanh(scores / TANH_SCALE)

    print(f"Total positions: {len(scores):,}")
    print(f"Diagnostic records: {diagnostic_count:,}")
    print(f"\nScore (centipawns) statistics:")
    print(f"  Min: {scores.min()}, Max: {scores.max()}")
    print(f"  Mean: {scores.mean():.1f}, Std: {scores.std():.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"\nTarget (tanh) statistics:")
    print(f"  Min: {targets.min():.4f}, Max: {targets.max():.4f}")
    print(f"  Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")

    if nn_type.upper() == "NNUE":
        stms = np.array([p['stm'] for p in positions])
        print(f"\nSTM distribution:")
        print(f"  White to move (stm=1): {(stms == 1).sum():,} ({100 * (stms == 1).mean():.1f}%)")
        print(f"  Black to move (stm=0): {(stms == 0).sum():,} ({100 * (stms == 0).mean():.1f}%)")

    # Check for anomalies
    print(f"\n=== Anomaly Check ===")
    extreme_scores = np.abs(scores) > 5000
    print(f"Positions with |score| > 5000: {extreme_scores.sum():,} ({100 * extreme_scores.mean():.2f}%)")

    positive_scores = (scores > 0).sum()
    negative_scores = (scores < 0).sum()
    print(f"Positive scores: {positive_scores:,} ({100 * positive_scores / len(scores):.1f}%)")
    print(f"Negative scores: {negative_scores:,} ({100 * negative_scores / len(scores):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Verify data correctness in DNN/NNUE shards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect and verify shards in data/ directory
    python verify_data.py

    # Verify specific DNN shard
    python verify_data.py --dnn-shard data/dnn/train_0001.bin.zst

    # Verify specific NNUE shard
    python verify_data.py --nnue-shard data/nnue/train_0001.bin.zst

    # Analyze shard statistics
    python verify_data.py --analyze data/nnue/train_0001.bin.zst --nn-type NNUE
"""
    )

    parser.add_argument('--dnn-shard', type=str, help='Path to DNN shard to verify')
    parser.add_argument('--nnue-shard', type=str, help='Path to NNUE shard to verify')
    parser.add_argument('--analyze', type=str, help='Path to shard to analyze statistics')
    parser.add_argument('--nn-type', type=str, choices=['DNN', 'NNUE'], help='NN type for --analyze')
    parser.add_argument('--data-dir', type=str, default='data', help='Base data directory (default: data)')
    parser.add_argument('--max-records', type=int, default=10, help='Max diagnostic records to verify (default: 10)')

    args = parser.parse_args()

    all_passed = True

    # If specific shards provided, verify them
    if args.dnn_shard:
        if not os.path.exists(args.dnn_shard):
            print(f"ERROR: DNN shard not found: {args.dnn_shard}")
            sys.exit(1)
        if not verify_dnn_shard(args.dnn_shard, args.max_records):
            all_passed = False

    if args.nnue_shard:
        if not os.path.exists(args.nnue_shard):
            print(f"ERROR: NNUE shard not found: {args.nnue_shard}")
            sys.exit(1)
        if not verify_nnue_shard(args.nnue_shard, args.max_records):
            all_passed = False

    if args.analyze:
        if not os.path.exists(args.analyze):
            print(f"ERROR: Shard not found: {args.analyze}")
            sys.exit(1)
        if not args.nn_type:
            # Try to infer from path
            if 'dnn' in args.analyze.lower():
                args.nn_type = 'DNN'
            elif 'nnue' in args.analyze.lower():
                args.nn_type = 'NNUE'
            else:
                print("ERROR: --nn-type required for --analyze")
                sys.exit(1)
        analyze_shard(args.analyze, args.nn_type)
        return

    # If no specific shards, auto-detect
    if not args.dnn_shard and not args.nnue_shard and not args.analyze:
        print(f"Searching for shards in {args.data_dir}/...")
        dnn_shards, nnue_shards = find_shards(args.data_dir)

        if not dnn_shards and not nnue_shards:
            print(f"No shard files found in {args.data_dir}/")
            print("Expected: data/dnn/*.bin.zst and/or data/nnue/*.bin.zst")
            sys.exit(1)

        if dnn_shards:
            print(f"\nFound {len(dnn_shards)} DNN shard(s)")
            if not verify_dnn_shard(dnn_shards[0], args.max_records):
                all_passed = False

        if nnue_shards:
            print(f"\nFound {len(nnue_shards)} NNUE shard(s)")
            if not verify_nnue_shard(nnue_shards[0], args.max_records):
                all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("✗ VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()