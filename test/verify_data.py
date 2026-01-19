#!/usr/bin/env python3
"""
Diagnostic script to verify data correctness between prepare_data.py and nn_train.py
"""

import io
import struct
import sys
import glob
import numpy as np
import zstandard as zstd

# Check what TANH_SCALE is set to
try:
    from nn_inference import TANH_SCALE, MAX_SCORE

    print(f"TANH_SCALE = {TANH_SCALE}")
    print(f"MAX_SCORE = {MAX_SCORE}")
except ImportError:
    print("Could not import from nn_inference, using defaults")
    TANH_SCALE = 400
    MAX_SCORE = 10000


def read_nnue_shard_sample(shard_path, max_positions=10):
    """Read a few positions from an NNUE shard for inspection."""
    dctx = zstd.ZstdDecompressor()

    with open(shard_path, 'rb') as f:
        reader = dctx.stream_reader(f)
        data = reader.read()
        reader.close()

    buf = io.BytesIO(data)
    positions = []

    while len(positions) < max_positions:
        score_bytes = buf.read(2)
        if len(score_bytes) < 2:
            break
        score_cp = struct.unpack('<h', score_bytes)[0]
        stm = struct.unpack('<B', buf.read(1))[0]

        num_white = struct.unpack('<B', buf.read(1))[0]
        white_features = []
        for _ in range(num_white):
            white_features.append(struct.unpack('<H', buf.read(2))[0])

        num_black = struct.unpack('<B', buf.read(1))[0]
        black_features = []
        for _ in range(num_black):
            black_features.append(struct.unpack('<H', buf.read(2))[0])

        positions.append({
            'score_cp': score_cp,
            'stm': stm,
            'white_features': white_features,
            'black_features': black_features
        })

    return positions


def analyze_shard(shard_path):
    """Analyze a full shard for statistics."""
    dctx = zstd.ZstdDecompressor()

    with open(shard_path, 'rb') as f:
        reader = dctx.stream_reader(f)
        data = reader.read()
        reader.close()

    buf = io.BytesIO(data)

    scores = []
    stms = []
    white_counts = []
    black_counts = []

    while True:
        score_bytes = buf.read(2)
        if len(score_bytes) < 2:
            break
        score_cp = struct.unpack('<h', score_bytes)[0]
        stm = struct.unpack('<B', buf.read(1))[0]

        num_white = struct.unpack('<B', buf.read(1))[0]
        buf.read(num_white * 2)  # Skip white features

        num_black = struct.unpack('<B', buf.read(1))[0]
        buf.read(num_black * 2)  # Skip black features

        scores.append(score_cp)
        stms.append(stm)
        white_counts.append(num_white)
        black_counts.append(num_black)

    scores = np.array(scores)
    stms = np.array(stms)
    white_counts = np.array(white_counts)
    black_counts = np.array(black_counts)

    # Compute tanh targets
    targets = np.tanh(scores / TANH_SCALE)

    print(f"\n=== Shard Analysis: {shard_path} ===")
    print(f"Total positions: {len(scores):,}")
    print(f"\nScore (centipawns) statistics:")
    print(f"  Min: {scores.min()}, Max: {scores.max()}")
    print(f"  Mean: {scores.mean():.1f}, Std: {scores.std():.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"\nTarget (tanh) statistics:")
    print(f"  Min: {targets.min():.4f}, Max: {targets.max():.4f}")
    print(f"  Mean: {targets.mean():.4f}, Std: {targets.std():.4f}")
    print(f"\nSTM distribution:")
    print(f"  White to move (stm=1): {(stms == 1).sum():,} ({100 * (stms == 1).mean():.1f}%)")
    print(f"  Black to move (stm=0): {(stms == 0).sum():,} ({100 * (stms == 0).mean():.1f}%)")
    print(f"\nFeature counts:")
    print(f"  White features: min={white_counts.min()}, max={white_counts.max()}, mean={white_counts.mean():.1f}")
    print(f"  Black features: min={black_counts.min()}, max={black_counts.max()}, mean={black_counts.mean():.1f}")

    # Check for anomalies
    print(f"\n=== Anomaly Check ===")
    extreme_scores = np.abs(scores) > 5000
    print(f"Positions with |score| > 5000: {extreme_scores.sum():,} ({100 * extreme_scores.mean():.2f}%)")

    # Check if scores look reasonable (should be roughly balanced around 0 for random positions)
    positive_scores = (scores > 0).sum()
    negative_scores = (scores < 0).sum()
    print(f"Positive scores: {positive_scores:,} ({100 * positive_scores / len(scores):.1f}%)")
    print(f"Negative scores: {negative_scores:,} ({100 * negative_scores / len(scores):.1f}%)")

    return {
        'scores': scores,
        'targets': targets,
        'stms': stms
    }


def main():
    if len(sys.argv) < 2:
        # Try to find shards automatically
        patterns = ['data/nnue/*.bin.zst', 'data/*.bin.zst']
        shards = []
        for pattern in patterns:
            shards.extend(glob.glob(pattern))

        if not shards:
            print("Usage: python verify_data.py <shard_path>")
            print("Or run from directory with data/nnue/*.bin.zst files")
            sys.exit(1)

        shard_path = sorted(shards)[0]
        print(f"Found shard: {shard_path}")
    else:
        shard_path = sys.argv[1]

    # Sample a few positions
    print("=== Sample Positions ===")
    positions = read_nnue_shard_sample(shard_path, max_positions=5)
    for i, pos in enumerate(positions):
        target = np.tanh(pos['score_cp'] / TANH_SCALE)
        print(f"\nPosition {i + 1}:")
        print(f"  Score (cp): {pos['score_cp']} -> target: {target:.4f}")
        print(f"  STM: {'White' if pos['stm'] == 1 else 'Black'} ({pos['stm']})")
        print(f"  White features ({len(pos['white_features'])}): {pos['white_features'][:5]}...")
        print(f"  Black features ({len(pos['black_features'])}): {pos['black_features'][:5]}...")

    # Full analysis
    analyze_shard(shard_path)

    # Expected loss calculation
    print("\n=== Expected Loss Analysis ===")
    print("For a random predictor outputting 0:")
    data = analyze_shard(shard_path)
    targets = data['targets']
    random_loss = np.mean(targets ** 2)
    print(f"  MSE loss (predict 0): {random_loss:.4f}")
    print(f"\nFor a predictor outputting mean target:")
    mean_target = targets.mean()
    mean_loss = np.mean((targets - mean_target) ** 2)
    print(f"  Mean target: {mean_target:.4f}")
    print(f"  MSE loss (predict mean): {mean_loss:.4f}")


if __name__ == "__main__":
    main()