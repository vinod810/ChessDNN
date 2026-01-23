#!/usr/bin/env python3
"""
nn_train.py - Train NNUE or DNN models from pre-processed binary shards.

MEMORY-OPTIMIZED VERSION:
- Uses NumPy arrays instead of Python dicts/lists for positions
- Streaming decompression to avoid loading entire shards into memory
- Reduced buffer sizes and worker counts
- Position-count-based queue limits

This script reads binary shard files created by prepare_data.py and trains
neural network models for chess position evaluation.

Usage:
    python nn_train.py --nn-type NNUE --data-dir data/nnue
    python nn_train.py --nn-type DNN --data-dir data/dnn --resume model/dnn.pt

Features:
    - Efficient sparse feature handling using index-based accumulation
    - Parallel data loading without blocking training
    - Prefetch queue for efficient GPU utilization
    - Shuffling across multiple shards for better training
    - 90/10 train/validation split by shard files
    - Learning rate scheduling and early stopping
    - Handles diagnostic records (marker=0xFF) with embedded FEN strings

Binary shard format (created by prepare_data.py):
    DNN Normal:     [score:int16][num_features:uint8][features:uint16[]]
    DNN Diagnostic: [0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[]]
                    [fen_length:uint8][fen_bytes]

    NNUE Normal:     [score:int16][stm:uint8][num_white:uint8][white:uint16[]]
                     [num_black:uint8][black:uint16[]]
    NNUE Diagnostic: [0xFF][score:int16][stm:uint8][num_white:uint8][white:uint16[]]
                     [num_black:uint8][black:uint16[]][fen_length:uint8][fen_bytes]
"""

import argparse
import os
import sys
import random
import time
import gc
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator
from collections import deque
import threading
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zstandard as zstd

# Import network architectures and constants from nn_inference
from nn_inference import (
    NNUE_INPUT_SIZE, DNN_INPUT_SIZE, NNUE_HIDDEN_SIZE, DNN_HIDDEN_LAYERS,
    TANH_SCALE, MAX_SCORE
)

# Training configuration
VALIDATION_SPLIT_RATIO = 0.01
BATCH_SIZE = 16384 # SF 16384
LEARNING_RATE_DENSE = 8.75e-4  # SF 8.75e-4
LEARNING_RATE = 8.75e-4 # SF 8.75e-4
# Embedding mode: True = use EmbeddingBag (sparse), False = use one-hot dense vectors
EMBEDDING_BAG = False
# Adjusted learning rate for dense mode (one-hot vectors may need different tuning)
POSITIONS_PER_EPOCH = 100_000_000  # SF 100_000_000
VALIDATION_SIZE = 1_000_000  # SF 1_000_000
EPOCHS = 600 # SF 600
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3

# MEMORY OPTIMIZATION: Reduced buffer sizes
SHARDS_PER_BUFFER = 2  # Reduced from 5 - Number of shards to hold in queue (controls memory)
PREFETCH_BATCHES = 2  # Number of batches to prefetch

# Checkpoint configuration
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs
VAL_INTERVAL = 1  # Validate every N epochs

# Garbage collection
GC_INTERVAL = 500  # Run GC every N batches (more frequent)

# Maximum features per position (for padding)
MAX_FEATURES_PER_POSITION = 32  # Chess has max 30 non-king pieces

# =============================================================================
# Sparse-Efficient Network Implementations for Training
# =============================================================================

class NNUENetworkSparse(nn.Module):
    """
    NNUE Network optimized for sparse training.

    Uses EmbeddingBag for efficient sparse feature accumulation in the first layer.
    This avoids creating huge dense tensors (batch_size × 40960).

    The architecture matches NNUENetwork from nn_inference.py:
    - Input: 40960 sparse features -> 256 hidden (via EmbeddingBag)
    - Hidden: 512 (concatenated white+black) -> 32 -> 32 -> 1
    """

    def __init__(self, input_size=NNUE_INPUT_SIZE, hidden_size=NNUE_HIDDEN_SIZE):
        super(NNUENetworkSparse, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer as EmbeddingBag (sparse-efficient)
        # EmbeddingBag computes: sum of embeddings for given indices + bias
        self.ft_weight = nn.EmbeddingBag(input_size, hidden_size, mode='sum', sparse=True)
        self.ft_bias = nn.Parameter(torch.zeros(hidden_size))

        # Remaining layers (dense, same as original)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_indices, white_offsets, black_indices, black_offsets, stm):
        """
        Forward pass with sparse indices.

        Args:
            white_indices: 1D tensor of all white feature indices (concatenated)
            white_offsets: 1D tensor of offsets into white_indices for each sample
            black_indices: 1D tensor of all black feature indices (concatenated)
            black_offsets: 1D tensor of offsets into black_indices for each sample
            stm: (batch_size, 1) tensor, 1.0 for white to move, 0.0 for black

        Returns:
            (batch_size, 1) output tensor
        """
        # Compute first layer activations using EmbeddingBag
        white_hidden = self.ft_weight(white_indices, white_offsets) + self.ft_bias
        black_hidden = self.ft_weight(black_indices, black_offsets) + self.ft_bias

        # Clipped ReLU [0, 1]
        white_hidden = torch.clamp(white_hidden, 0, 1)
        black_hidden = torch.clamp(black_hidden, 0, 1)

        # Concatenate based on side to move
        # stm=1 (white): [white, black], stm=0 (black): [black, white]
        stm_expanded = stm.unsqueeze(-1) if stm.dim() == 1 else stm
        hidden = torch.where(
            stm_expanded.bool(),
            torch.cat([white_hidden, black_hidden], dim=-1),
            torch.cat([black_hidden, white_hidden], dim=-1)
        )

        # Dense layers with clipped ReLU
        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return x

    def to_inference_model(self):
        """
        Convert to the standard NNUENetwork format for inference.
        Returns a state dict compatible with nn_inference.NNUENetwork.
        """
        state_dict = {
            'ft.weight': self.ft_weight.weight.data.t(),  # EmbeddingBag is (num_emb, emb_dim), Linear is (out, in)
            'ft.bias': self.ft_bias.data,
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
        }
        return state_dict

    def load_from_inference_model(self, state_dict):
        """Load weights from standard NNUENetwork state dict."""
        self.ft_weight.weight.data = state_dict['ft.weight'].t()
        self.ft_bias.data = state_dict['ft.bias']
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']


class DNNNetworkSparse(nn.Module):
    """
    DNN Network optimized for sparse training.

    Uses EmbeddingBag for efficient sparse feature accumulation in the first layer.

    The architecture matches DNNNetwork from nn_inference.py:
    - Input: 768 sparse features -> 1024 hidden (via EmbeddingBag)
    - Hidden: 1024 -> 256 -> 32 -> 1
    """

    def __init__(self, input_size=DNN_INPUT_SIZE, hidden_layers=None):
        super(DNNNetworkSparse, self).__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # First layer as EmbeddingBag (sparse-efficient)
        self.l1_weight = nn.EmbeddingBag(input_size, hidden_layers[0], mode='sum', sparse=True)
        self.l1_bias = nn.Parameter(torch.zeros(hidden_layers[0]))

        # Remaining layers (dense)
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, indices, offsets):
        """
        Forward pass with sparse indices.

        Args:
            indices: 1D tensor of all feature indices (concatenated)
            offsets: 1D tensor of offsets into indices for each sample

        Returns:
            (batch_size, 1) output tensor
        """
        # Compute first layer activation using EmbeddingBag
        x = self.l1_weight(indices, offsets) + self.l1_bias
        x = torch.clamp(x, 0, 1)

        # Dense layers with clipped ReLU
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.clamp(self.l3(x), 0, 1)
        x = self.l4(x)
        return x

    def to_inference_model(self):
        """
        Convert to the standard DNNNetwork format for inference.
        Returns a state dict compatible with nn_inference.DNNNetwork.
        """
        state_dict = {
            'l1.weight': self.l1_weight.weight.data.t(),
            'l1.bias': self.l1_bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
            'l4.weight': self.l4.weight.data,
            'l4.bias': self.l4.bias.data,
        }
        return state_dict

    def load_from_inference_model(self, state_dict):
        """Load weights from standard DNNNetwork state dict."""
        self.l1_weight.weight.data = state_dict['l1.weight'].t()
        self.l1_bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']
        self.l4.weight.data = state_dict['l4.weight']
        self.l4.bias.data = state_dict['l4.bias']


# =============================================================================
# Dense Network Implementations (using one-hot vectors instead of EmbeddingBag)
# =============================================================================

class NNUENetworkDense(nn.Module):
    """
    NNUE Network using dense one-hot vectors instead of EmbeddingBag.

    This creates explicit one-hot dense vectors and uses standard Linear layers
    for the first layer. More memory intensive but may train differently.
    """

    def __init__(self, input_size=NNUE_INPUT_SIZE, hidden_size=NNUE_HIDDEN_SIZE):
        super(NNUENetworkDense, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer as standard Linear (for dense one-hot input)
        self.ft = nn.Linear(input_size, hidden_size)

        # Remaining layers (dense, same as original)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_onehot, black_onehot, stm):
        """
        Forward pass with dense one-hot vectors.

        Args:
            white_onehot: (batch_size, input_size) dense one-hot tensor
            black_onehot: (batch_size, input_size) dense one-hot tensor
            stm: (batch_size, 1) tensor, 1.0 for white to move, 0.0 for black

        Returns:
            (batch_size, 1) output tensor
        """
        # Compute first layer activations
        white_hidden = self.ft(white_onehot)
        black_hidden = self.ft(black_onehot)

        # Clipped ReLU [0, 1]
        white_hidden = torch.clamp(white_hidden, 0, 1)
        black_hidden = torch.clamp(black_hidden, 0, 1)

        # Concatenate based on side to move
        stm_expanded = stm.unsqueeze(-1) if stm.dim() == 1 else stm
        hidden = torch.where(
            stm_expanded.bool(),
            torch.cat([white_hidden, black_hidden], dim=-1),
            torch.cat([black_hidden, white_hidden], dim=-1)
        )

        # Dense layers with clipped ReLU
        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return x

    def to_inference_model(self):
        """Convert to inference format."""
        state_dict = {
            'ft.weight': self.ft.weight.data,
            'ft.bias': self.ft.bias.data,
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
        }
        return state_dict

    def load_from_inference_model(self, state_dict):
        """Load weights from standard NNUENetwork state dict."""
        self.ft.weight.data = state_dict['ft.weight']
        self.ft.bias.data = state_dict['ft.bias']
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']


class DNNNetworkDense(nn.Module):
    """
    DNN Network using dense one-hot vectors instead of EmbeddingBag.

    Uses standard Linear layers throughout.
    """

    def __init__(self, input_size=DNN_INPUT_SIZE, hidden_layers=None):
        super(DNNNetworkDense, self).__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # All layers as standard Linear
        self.l1 = nn.Linear(input_size, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, onehot):
        """
        Forward pass with dense one-hot vector.

        Args:
            onehot: (batch_size, input_size) dense one-hot tensor

        Returns:
            (batch_size, 1) output tensor
        """
        x = self.l1(onehot)
        x = torch.clamp(x, 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.clamp(self.l3(x), 0, 1)
        x = self.l4(x)
        return x

    def to_inference_model(self):
        """Convert to inference format."""
        state_dict = {
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
            'l4.weight': self.l4.weight.data,
            'l4.bias': self.l4.bias.data,
        }
        return state_dict

    def load_from_inference_model(self, state_dict):
        """Load weights from standard DNNNetwork state dict."""
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']
        self.l4.weight.data = state_dict['l4.weight']
        self.l4.bias.data = state_dict['l4.bias']


# =============================================================================
# Memory-Efficient Data Structures
# =============================================================================

class DNNPositionArray:
    """
    Memory-efficient storage for DNN positions using NumPy arrays.

    Instead of storing each position as a Python dict with lists,
    we use compact NumPy arrays:
    - scores: int16 array of scores
    - features: uint16 array of all features concatenated
    - offsets: int32 array of starting offsets for each position's features

    Memory comparison for 1M positions with avg 20 features:
    - Python dicts: ~300 bytes/position = 300 MB
    - NumPy arrays: ~42 bytes/position = 42 MB (7x reduction)
    """
    __slots__ = ['scores', 'features', 'offsets', 'num_positions']

    def __init__(self, scores: np.ndarray, features: np.ndarray, offsets: np.ndarray):
        self.scores = scores  # int16[num_positions]
        self.features = features  # uint16[total_features]
        self.offsets = offsets  # int32[num_positions + 1] - last element is total feature count
        self.num_positions = len(scores)

    def __len__(self):
        return self.num_positions

    def get_position(self, idx: int) -> Tuple[int, np.ndarray]:
        """Get a single position's score and features."""
        score = self.scores[idx]
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        features = self.features[start:end]
        return score, features

    def get_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a batch of positions efficiently.

        Returns:
            scores: int16[batch_size]
            all_features: uint16[total_features_in_batch]
            batch_offsets: int32[batch_size] - offsets into all_features
        """
        batch_size = len(indices)
        scores = self.scores[indices]

        # Calculate total features needed
        starts = self.offsets[indices]
        ends = self.offsets[indices + 1]
        lengths = ends - starts
        total_features = lengths.sum()

        # Allocate output arrays
        all_features = np.empty(total_features, dtype=np.uint16)
        batch_offsets = np.empty(batch_size, dtype=np.int32)

        # Copy features
        out_pos = 0
        for i, (start, length) in enumerate(zip(starts, lengths)):
            batch_offsets[i] = out_pos
            all_features[out_pos:out_pos + length] = self.features[start:start + length]
            out_pos += length

        return scores, all_features, batch_offsets


class NNUEPositionArray:
    """
    Memory-efficient storage for NNUE positions using NumPy arrays.

    Memory comparison for 1M positions with avg 16 white + 16 black features:
    - Python dicts: ~400 bytes/position = 400 MB
    - NumPy arrays: ~70 bytes/position = 70 MB (5.7x reduction)
    """
    __slots__ = ['scores', 'stm', 'white_features', 'white_offsets',
                 'black_features', 'black_offsets', 'num_positions']

    def __init__(self, scores: np.ndarray, stm: np.ndarray,
                 white_features: np.ndarray, white_offsets: np.ndarray,
                 black_features: np.ndarray, black_offsets: np.ndarray):
        self.scores = scores  # int16[num_positions]
        self.stm = stm  # uint8[num_positions]
        self.white_features = white_features  # uint16[total_white_features]
        self.white_offsets = white_offsets  # int32[num_positions + 1]
        self.black_features = black_features  # uint16[total_black_features]
        self.black_offsets = black_offsets  # int32[num_positions + 1]
        self.num_positions = len(scores)

    def __len__(self):
        return self.num_positions

    def get_batch(self, indices: np.ndarray):
        """
        Get a batch of positions efficiently.

        Returns:
            scores, stm, white_features, white_batch_offsets, black_features, black_batch_offsets
        """
        batch_size = len(indices)
        scores = self.scores[indices]
        stm = self.stm[indices]

        # White features
        w_starts = self.white_offsets[indices]
        w_ends = self.white_offsets[indices + 1]
        w_lengths = w_ends - w_starts
        total_white = w_lengths.sum()

        all_white = np.empty(total_white, dtype=np.uint16)
        white_batch_offsets = np.empty(batch_size, dtype=np.int32)

        out_pos = 0
        for i, (start, length) in enumerate(zip(w_starts, w_lengths)):
            white_batch_offsets[i] = out_pos
            all_white[out_pos:out_pos + length] = self.white_features[start:start + length]
            out_pos += length

        # Black features
        b_starts = self.black_offsets[indices]
        b_ends = self.black_offsets[indices + 1]
        b_lengths = b_ends - b_starts
        total_black = b_lengths.sum()

        all_black = np.empty(total_black, dtype=np.uint16)
        black_batch_offsets = np.empty(batch_size, dtype=np.int32)

        out_pos = 0
        for i, (start, length) in enumerate(zip(b_starts, b_lengths)):
            black_batch_offsets[i] = out_pos
            all_black[out_pos:out_pos + length] = self.black_features[start:start + length]
            out_pos += length

        return scores, stm, all_white, white_batch_offsets, all_black, black_batch_offsets


# =============================================================================
# Data Loading (Memory Optimized)
# =============================================================================

class ShardReader:
    """
    Reads positions from compressed binary shard files.

    MEMORY OPTIMIZATION: Uses shard_io for parsing, then converts to
    compact NumPy-based position arrays for efficient training.
    """

    def __init__(self, nn_type: str):
        self.nn_type = nn_type.upper()
        # Import here to avoid circular imports
        from shard_io import ShardReader as BaseShardReader
        self._base_reader = BaseShardReader(nn_type)

    def read_shard(self, shard_path: str):
        """
        Read all positions from a shard file into memory-efficient arrays.

        Returns:
            DNNPositionArray or NNUEPositionArray depending on nn_type
        """
        # Use shard_io to read positions (handles diagnostic records properly)
        positions = self._base_reader.read_all_positions(shard_path, include_fen=False, skip_diagnostic=True)

        if self.nn_type == "DNN":
            return self._convert_dnn_positions(positions)
        else:
            return self._convert_nnue_positions(positions)

    def _convert_dnn_positions(self, positions: list) -> DNNPositionArray:
        """Convert list of position dicts to compact DNN arrays."""
        num_positions = len(positions)
        scores = np.empty(num_positions, dtype=np.int16)
        offsets = np.empty(num_positions + 1, dtype=np.int32)

        # Calculate total features
        total_features = sum(len(p['features']) for p in positions)
        all_features = np.empty(total_features, dtype=np.uint16)

        offset = 0
        for i, pos in enumerate(positions):
            scores[i] = pos['score_cp']
            offsets[i] = offset
            features = pos['features']
            for j, f in enumerate(features):
                all_features[offset + j] = f
            offset += len(features)
        offsets[num_positions] = offset

        return DNNPositionArray(scores, all_features, offsets)

    def _convert_nnue_positions(self, positions: list) -> NNUEPositionArray:
        """Convert list of position dicts to compact NNUE arrays."""
        num_positions = len(positions)
        scores = np.empty(num_positions, dtype=np.int16)
        stm_arr = np.empty(num_positions, dtype=np.uint8)
        white_offsets = np.empty(num_positions + 1, dtype=np.int32)
        black_offsets = np.empty(num_positions + 1, dtype=np.int32)

        total_white = sum(len(p['white_features']) for p in positions)
        total_black = sum(len(p['black_features']) for p in positions)
        all_white = np.empty(total_white, dtype=np.uint16)
        all_black = np.empty(total_black, dtype=np.uint16)

        w_offset = 0
        b_offset = 0
        for i, pos in enumerate(positions):
            scores[i] = pos['score_cp']
            stm_arr[i] = pos['stm']

            white_offsets[i] = w_offset
            white_feat = pos['white_features']
            for j, f in enumerate(white_feat):
                all_white[w_offset + j] = f
            w_offset += len(white_feat)

            black_offsets[i] = b_offset
            black_feat = pos['black_features']
            for j, f in enumerate(black_feat):
                all_black[b_offset + j] = f
            b_offset += len(black_feat)

        white_offsets[num_positions] = w_offset
        black_offsets[num_positions] = b_offset

        return NNUEPositionArray(scores, stm_arr, all_white, white_offsets, all_black, black_offsets)


def create_dnn_batch_from_array(
        position_array: DNNPositionArray,
        indices: np.ndarray,
        device: torch.device
):
    """
    Create sparse batch tensors for DNN training from position array.
    """
    scores, all_features, batch_offsets = position_array.get_batch(indices)

    # Compute targets
    targets = np.tanh(scores.astype(np.float32) / TANH_SCALE)

    # Convert to tensors
    indices_t = torch.from_numpy(all_features.astype(np.int64)).to(device)
    offsets_t = torch.from_numpy(batch_offsets.astype(np.int64)).to(device)
    targets_t = torch.from_numpy(targets).to(device).unsqueeze(1)

    return indices_t, offsets_t, targets_t


def create_nnue_batch_from_array(
        position_array: NNUEPositionArray,
        indices: np.ndarray,
        device: torch.device
):
    """
    Create sparse batch tensors for NNUE training from position array.
    """
    scores, stm, white_feat, white_off, black_feat, black_off = position_array.get_batch(indices)

    # Compute targets
    targets = np.tanh(scores.astype(np.float32) / TANH_SCALE)

    # Convert to tensors
    white_indices = torch.from_numpy(white_feat.astype(np.int64)).to(device)
    white_offsets = torch.from_numpy(white_off.astype(np.int64)).to(device)
    black_indices = torch.from_numpy(black_feat.astype(np.int64)).to(device)
    black_offsets = torch.from_numpy(black_off.astype(np.int64)).to(device)
    stm_t = torch.from_numpy(stm.astype(np.float32)).to(device).unsqueeze(1)
    targets_t = torch.from_numpy(targets).to(device).unsqueeze(1)

    return white_indices, white_offsets, black_indices, black_offsets, stm_t, targets_t


def create_dnn_batch_from_array_dense(
        position_array: DNNPositionArray,
        indices: np.ndarray,
        device: torch.device,
        input_size: int = DNN_INPUT_SIZE
):
    """
    Create dense one-hot batch tensors for DNN training from position array.
    """
    scores, all_features, batch_offsets = position_array.get_batch(indices)
    batch_size = len(indices)

    # Compute targets
    targets = np.tanh(scores.astype(np.float32) / TANH_SCALE)

    # Create dense one-hot tensor
    onehot = torch.zeros(batch_size, input_size, device=device)

    # Fill in the one-hot vectors
    for i in range(batch_size):
        start = batch_offsets[i]
        end = batch_offsets[i + 1] if i + 1 < batch_size else len(all_features)
        feature_indices = all_features[start:end]
        onehot[i, feature_indices] = 1.0

    targets_t = torch.from_numpy(targets).to(device).unsqueeze(1)

    return onehot, targets_t


def create_nnue_batch_from_array_dense(
        position_array: NNUEPositionArray,
        indices: np.ndarray,
        device: torch.device,
        input_size: int = NNUE_INPUT_SIZE
):
    """
    Create dense one-hot batch tensors for NNUE training from position array.
    """
    scores, stm, white_feat, white_off, black_feat, black_off = position_array.get_batch(indices)
    batch_size = len(indices)

    # Compute targets
    targets = np.tanh(scores.astype(np.float32) / TANH_SCALE)

    # Create dense one-hot tensors for white and black
    white_onehot = torch.zeros(batch_size, input_size, device=device)
    black_onehot = torch.zeros(batch_size, input_size, device=device)

    # Fill in the one-hot vectors for white
    for i in range(batch_size):
        start = white_off[i]
        end = white_off[i + 1] if i + 1 < batch_size else len(white_feat)
        feature_indices = white_feat[start:end]
        white_onehot[i, feature_indices] = 1.0

    # Fill in the one-hot vectors for black
    for i in range(batch_size):
        start = black_off[i]
        end = black_off[i + 1] if i + 1 < batch_size else len(black_feat)
        feature_indices = black_feat[start:end]
        black_onehot[i, feature_indices] = 1.0

    stm_t = torch.from_numpy(stm.astype(np.float32)).to(device).unsqueeze(1)
    targets_t = torch.from_numpy(targets).to(device).unsqueeze(1)

    return white_onehot, black_onehot, stm_t, targets_t


class DataLoader:
    """
    Memory-efficient data loader with parallel shard reading and batch preparation.

    MEMORY OPTIMIZATIONS:
    - Uses NumPy-based position arrays instead of Python dicts
    - Queue size limits control memory (shards_per_buffer shards max in queue)
    - Reduced default worker count
    """

    def __init__(
            self,
            shard_files: List[str],
            nn_type: str,
            batch_size: int,
            device: torch.device,
            shards_per_buffer: int = SHARDS_PER_BUFFER,
            prefetch_batches: int = PREFETCH_BATCHES,
            shuffle: bool = True,
            max_positions: Optional[int] = None,
            num_workers: int = 2,  # Reduced default from 4
            use_embedding_bag: bool = EMBEDDING_BAG
    ):
        self.shard_files = shard_files.copy()
        self.nn_type = nn_type.upper()
        self.batch_size = batch_size
        self.device = device
        self.shards_per_buffer = shards_per_buffer
        self.prefetch_batches = prefetch_batches
        self.shuffle = shuffle
        self.max_positions = max_positions
        self.num_workers = num_workers
        self.use_embedding_bag = use_embedding_bag

        # Queues for inter-thread communication
        self.shard_queue = queue.Queue()  # Shards to be read
        self.position_queue = queue.Queue(maxsize=shards_per_buffer)  # Loaded position arrays
        self.batch_queue = queue.Queue(maxsize=prefetch_batches)  # Ready batches

        self.stop_event = threading.Event()
        self.reader_threads = []
        self.batcher_thread = None
        self.positions_yielded = 0
        self._positions_yielded_lock = threading.Lock()

    def _reader_worker(self, worker_id: int):
        """Worker thread that reads shards from shard_queue into position_queue."""
        reader = ShardReader(self.nn_type)

        while not self.stop_event.is_set():
            try:
                shard_path = self.shard_queue.get(timeout=0.5)
                if shard_path is None:  # Poison pill
                    break

                try:
                    position_array = reader.read_shard(shard_path)

                    # Block until there's space in the queue (queue maxsize controls buffering)
                    while not self.stop_event.is_set():
                        try:
                            self.position_queue.put(position_array, timeout=1.0)
                            break
                        except queue.Full:
                            continue  # Keep trying

                except Exception as e:
                    print(f"\nWarning: Worker {worker_id} error reading {shard_path}: {e}")
                    traceback.print_exc()

            except queue.Empty:
                continue

        # Signal batcher that this reader is done
        while not self.stop_event.is_set():
            try:
                self.position_queue.put(None, timeout=1.0)
                break
            except queue.Full:
                continue

    def _batcher_worker(self):
        """Worker thread that creates batches from loaded position arrays."""
        # Buffer holds [position_array, shuffled_indices, current_index]
        buffer_arrays = []
        active_readers = self.num_workers
        batches_created = 0

        def get_total_available():
            """Calculate total positions available in buffer."""
            return sum(len(item[1]) - item[2] for item in buffer_arrays)

        while not self.stop_event.is_set():
            # Check if we've reached max positions
            with self._positions_yielded_lock:
                if self.max_positions and self.positions_yielded >= self.max_positions:
                    break

            total_available = get_total_available()

            # Try to get more position arrays from readers if buffer is low
            if total_available < self.batch_size * 2 or len(buffer_arrays) == 0:
                try:
                    position_array = self.position_queue.get(timeout=0.5)
                    if position_array is None:  # Reader finished
                        active_readers -= 1
                        if active_readers <= 0 and get_total_available() < self.batch_size:
                            break
                        continue

                    # Create shuffled indices for this array
                    indices = np.arange(len(position_array), dtype=np.int32)
                    if self.shuffle:
                        np.random.shuffle(indices)

                    buffer_arrays.append([position_array, indices, 0])

                except queue.Empty:
                    if active_readers <= 0 and get_total_available() < self.batch_size:
                        break
                    continue  # Keep trying to get data

            total_available = get_total_available()

            # Create a batch if we have enough data
            if total_available >= self.batch_size:
                with self._positions_yielded_lock:
                    if self.max_positions and self.positions_yielded >= self.max_positions:
                        break

                # Gather batch_size positions from buffer arrays
                batch_indices_list = []
                batch_arrays_list = []
                remaining = self.batch_size

                i = 0
                while remaining > 0 and i < len(buffer_arrays):
                    arr, indices, pos = buffer_arrays[i]
                    available = len(indices) - pos
                    take = min(remaining, available)

                    batch_indices_list.append(indices[pos:pos + take])
                    batch_arrays_list.append(arr)

                    buffer_arrays[i][2] = pos + take
                    remaining -= take

                    if buffer_arrays[i][2] >= len(indices):
                        # This array is exhausted - remove it
                        buffer_arrays.pop(i)
                        # Don't increment i since we removed the element
                    else:
                        i += 1

                if remaining > 0:
                    # Not enough data (shouldn't happen)
                    continue

                try:
                    # Create batch
                    if len(batch_arrays_list) == 1:
                        # Simple case: all from one array
                        if self.nn_type == "DNN":
                            if self.use_embedding_bag:
                                batch = create_dnn_batch_from_array(
                                    batch_arrays_list[0], batch_indices_list[0], self.device
                                )
                            else:
                                batch = create_dnn_batch_from_array_dense(
                                    batch_arrays_list[0], batch_indices_list[0], self.device
                                )
                        else:
                            if self.use_embedding_bag:
                                batch = create_nnue_batch_from_array(
                                    batch_arrays_list[0], batch_indices_list[0], self.device
                                )
                            else:
                                batch = create_nnue_batch_from_array_dense(
                                    batch_arrays_list[0], batch_indices_list[0], self.device
                                )
                    else:
                        # Multiple arrays: need to combine batches
                        batch = self._combine_batches(batch_arrays_list, batch_indices_list)

                    # Use blocking put to ensure batch is delivered
                    self.batch_queue.put(batch)
                    batches_created += 1

                    with self._positions_yielded_lock:
                        self.positions_yielded += self.batch_size

                except Exception as e:
                    print(f"\nWarning: Error creating batch {batches_created}: {e}")
                    traceback.print_exc()

        # Signal end
        try:
            self.batch_queue.put(None, timeout=1.0)
        except:
            pass

    def _combine_batches(self, arrays_list, indices_list):
        """Combine positions from multiple arrays into one batch."""
        if self.nn_type == "DNN":
            all_scores = []
            all_features = []
            all_offsets = []
            current_offset = 0

            for arr, indices in zip(arrays_list, indices_list):
                scores, features, offsets = arr.get_batch(indices)
                all_scores.append(scores)
                all_features.append(features)
                # Adjust offsets
                all_offsets.append(offsets + current_offset)
                current_offset += len(features)

            scores = np.concatenate(all_scores)
            features = np.concatenate(all_features)
            offsets = np.concatenate(all_offsets)

            targets = np.tanh(scores.astype(np.float32) / TANH_SCALE)

            if self.use_embedding_bag:
                indices_t = torch.from_numpy(features.astype(np.int64)).to(self.device)
                offsets_t = torch.from_numpy(offsets.astype(np.int64)).to(self.device)
                targets_t = torch.from_numpy(targets).to(self.device).unsqueeze(1)
                return indices_t, offsets_t, targets_t
            else:
                # Dense mode: create one-hot vectors
                batch_size = sum(len(idx) for idx in indices_list)
                onehot = torch.zeros(batch_size, DNN_INPUT_SIZE, device=self.device)
                for i in range(batch_size):
                    start = offsets[i]
                    end = offsets[i + 1] if i + 1 < batch_size else len(features)
                    feature_indices = features[start:end]
                    onehot[i, feature_indices] = 1.0
                targets_t = torch.from_numpy(targets).to(self.device).unsqueeze(1)
                return onehot, targets_t
        else:
            # NNUE
            all_scores = []
            all_stm = []
            all_white = []
            all_white_off = []
            all_black = []
            all_black_off = []
            w_offset = 0
            b_offset = 0

            for arr, indices in zip(arrays_list, indices_list):
                scores, stm, white_feat, white_off, black_feat, black_off = arr.get_batch(indices)
                all_scores.append(scores)
                all_stm.append(stm)
                all_white.append(white_feat)
                all_white_off.append(white_off + w_offset)
                w_offset += len(white_feat)
                all_black.append(black_feat)
                all_black_off.append(black_off + b_offset)
                b_offset += len(black_feat)

            scores = np.concatenate(all_scores)
            stm = np.concatenate(all_stm)
            white_feat = np.concatenate(all_white)
            white_off = np.concatenate(all_white_off)
            black_feat = np.concatenate(all_black)
            black_off = np.concatenate(all_black_off)

            targets = np.tanh(scores.astype(np.float32) / TANH_SCALE)

            if self.use_embedding_bag:
                white_indices = torch.from_numpy(white_feat.astype(np.int64)).to(self.device)
                white_offsets = torch.from_numpy(white_off.astype(np.int64)).to(self.device)
                black_indices = torch.from_numpy(black_feat.astype(np.int64)).to(self.device)
                black_offsets = torch.from_numpy(black_off.astype(np.int64)).to(self.device)
                stm_t = torch.from_numpy(stm.astype(np.float32)).to(self.device).unsqueeze(1)
                targets_t = torch.from_numpy(targets).to(self.device).unsqueeze(1)
                return white_indices, white_offsets, black_indices, black_offsets, stm_t, targets_t
            else:
                # Dense mode: create one-hot vectors
                batch_size = sum(len(idx) for idx in indices_list)
                white_onehot = torch.zeros(batch_size, NNUE_INPUT_SIZE, device=self.device)
                black_onehot = torch.zeros(batch_size, NNUE_INPUT_SIZE, device=self.device)
                for i in range(batch_size):
                    w_start = white_off[i]
                    w_end = white_off[i + 1] if i + 1 < batch_size else len(white_feat)
                    white_onehot[i, white_feat[w_start:w_end]] = 1.0
                    b_start = black_off[i]
                    b_end = black_off[i + 1] if i + 1 < batch_size else len(black_feat)
                    black_onehot[i, black_feat[b_start:b_end]] = 1.0
                stm_t = torch.from_numpy(stm.astype(np.float32)).to(self.device).unsqueeze(1)
                targets_t = torch.from_numpy(targets).to(self.device).unsqueeze(1)
                return white_onehot, black_onehot, stm_t, targets_t

    def start(self):
        """Start all worker threads."""
        self.stop_event.clear()
        with self._positions_yielded_lock:
            self.positions_yielded = 0

        # Prepare shard list
        shard_list = self.shard_files.copy()
        if self.shuffle:
            random.shuffle(shard_list)

        # Fill shard queue
        for shard in shard_list:
            self.shard_queue.put(shard)

        # Add poison pills for readers
        for _ in range(self.num_workers):
            self.shard_queue.put(None)

        # Start reader threads
        self.reader_threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._reader_worker, args=(i,), daemon=True)
            t.start()
            self.reader_threads.append(t)

        # Start batcher thread
        self.batcher_thread = threading.Thread(target=self._batcher_worker, daemon=True)
        self.batcher_thread.start()

    def stop(self):
        """Stop all worker threads."""
        self.stop_event.set()

        # Clear queues to unblock threads
        for _ in range(100):  # More aggressive clearing
            try:
                self.shard_queue.get_nowait()
            except queue.Empty:
                break

        for _ in range(100):
            try:
                self.position_queue.get_nowait()
            except queue.Empty:
                break

        for _ in range(100):
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for threads with longer timeout
        for t in self.reader_threads:
            t.join(timeout=5.0)
        if self.batcher_thread:
            self.batcher_thread.join(timeout=5.0)

        # Force garbage collection
        gc.collect()

    def __iter__(self):
        """Iterate over batches."""
        self.start()
        batch_count = 0
        try:
            while True:
                try:
                    batch = self.batch_queue.get(timeout=60.0)  # Increased timeout
                    if batch is None:
                        break
                    batch_count += 1
                    yield batch
                except queue.Empty:
                    # Check if batcher is still running
                    if self.batcher_thread and not self.batcher_thread.is_alive():
                        # Batcher died, check if there's still data
                        try:
                            batch = self.batch_queue.get_nowait()
                            if batch is not None:
                                batch_count += 1
                                yield batch
                                continue
                        except queue.Empty:
                            pass
                        break
                    print(
                        f"\nWarning: Batch queue timeout after {batch_count} batches, batcher alive: {self.batcher_thread.is_alive()}")
        finally:
            self.stop()

    def get_positions_yielded(self) -> int:
        with self._positions_yielded_lock:
            return self.positions_yielded


class Trainer:
    """Handles the training loop with validation and checkpointing."""

    def __init__(
            self,
            model: nn.Module,
            nn_type: str,
            data_dir: str,
            device: torch.device,
            batch_size: int = BATCH_SIZE,
            lr: float = LEARNING_RATE,
            num_workers: int = 2,  # Reduced default from 4
            use_embedding_bag: bool = EMBEDDING_BAG,
            val_ratio: float = VALIDATION_SPLIT_RATIO
    ):
        self.model = model.to(device)
        self.nn_type = nn_type.upper()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_embedding_bag = use_embedding_bag

        # Initial shard discovery
        self._refresh_shards()

        self.criterion = nn.MSELoss()

        # For sparse EmbeddingBag, we need to use SparseAdam for the sparse parameters
        # and regular Adam for dense parameters
        sparse_params = []
        dense_params = []
        for name, param in model.named_parameters():
            if 'weight' in name and hasattr(model, name.split('.')[0]):
                module = getattr(model, name.split('.')[0])
                if isinstance(module, nn.EmbeddingBag) and module.sparse:
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            else:
                dense_params.append(param)

        # Use SparseAdam for embedding parameters, Adam for the rest
        if sparse_params:
            self.optimizer = optim.SparseAdam(sparse_params, lr=lr)
            self.optimizer_dense = optim.Adam(dense_params, lr=lr)
        else:
            # Dense mode: use Adam for all parameters
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.optimizer_dense = None

        # Scheduler only on dense optimizer (or main optimizer if no sparse)
        main_optimizer = self.optimizer_dense if self.optimizer_dense else self.optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            main_optimizer, mode='min', factor=0.5, patience=LR_PATIENCE
        )

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

    def _refresh_shards(self):
        """Re-discover shards from data directory and split into train/val."""
        all_shards = discover_shards(self.data_dir, self.nn_type)
        self.train_shards, self.val_shards = split_shards(all_shards, val_ratio=self.val_ratio)
        return len(all_shards)

    def _forward_dnn(self, batch):
        """Forward pass for DNN."""
        if self.use_embedding_bag:
            # Sparse batch: (indices, offsets, targets)
            indices, offsets, targets = batch
            outputs = self.model(indices, offsets)
        else:
            # Dense batch: (onehot, targets)
            onehot, targets = batch
            outputs = self.model(onehot)
        return outputs, targets

    def _forward_nnue(self, batch):
        """Forward pass for NNUE."""
        if self.use_embedding_bag:
            # Sparse batch: (white_indices, white_offsets, black_indices, black_offsets, stm, targets)
            white_indices, white_offsets, black_indices, black_offsets, stm, targets = batch
            outputs = self.model(white_indices, white_offsets, black_indices, black_offsets, stm)
        else:
            # Dense batch: (white_onehot, black_onehot, stm, targets)
            white_onehot, black_onehot, stm, targets = batch
            outputs = self.model(white_onehot, black_onehot, stm)
        return outputs, targets

    def train_epoch(self, positions_per_epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Calculate report interval for ~10 updates per epoch
        expected_batches = positions_per_epoch // self.batch_size
        report_interval = max(1, expected_batches // 100)

        loader = DataLoader(
            self.train_shards,
            self.nn_type,
            self.batch_size,
            self.device,
            shuffle=True,
            max_positions=positions_per_epoch,
            num_workers=self.num_workers,
            use_embedding_bag=self.use_embedding_bag
        )

        start_time = time.time()

        for batch in loader:
            # Forward pass
            if self.nn_type == "DNN":
                outputs, targets = self._forward_dnn(batch)
            else:
                outputs, targets = self._forward_nnue(batch)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            if self.optimizer_dense:
                self.optimizer_dense.zero_grad()

            loss.backward()

            # Gradient clipping (only for dense parameters)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.grad is not None and not p.grad.is_sparse],
                1.0
            )

            self.optimizer.step()
            if self.optimizer_dense:
                self.optimizer_dense.step()

            total_loss += loss.item()
            num_batches += 1

            # Progress reporting (~10 times per epoch)
            if num_batches % report_interval == 0:
                pct = 100 * num_batches * self.batch_size / positions_per_epoch
                print(f"\r  [{pct:5.1f}%] Loss: {loss.item():.6f}", end='', flush=True)

            # Garbage collection (more frequent)
            if num_batches % GC_INTERVAL == 0:
                gc.collect()

        print()  # New line after progress

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def validate(self, max_positions: int = VALIDATION_SIZE) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        loader = DataLoader(
            self.val_shards,
            self.nn_type,
            self.batch_size,
            self.device,
            shuffle=False,
            max_positions=max_positions,
            num_workers=self.num_workers,
            use_embedding_bag=self.use_embedding_bag
        )

        with torch.no_grad():
            for batch in loader:
                if self.nn_type == "DNN":
                    outputs, targets = self._forward_dnn(batch)
                else:
                    outputs, targets = self._forward_nnue(batch)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint in inference-compatible format."""
        # Convert sparse training model to inference format
        inference_state_dict = self.model.to_inference_model()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': inference_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_dense_state_dict': self.optimizer_dense.state_dict() if self.optimizer_dense else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_val_loss,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf'),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else float('inf'),
            'nn_type': self.nn_type,
            'history': self.history
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Verify nn_type matches
        if checkpoint.get('nn_type', self.nn_type) != self.nn_type:
            raise ValueError(f"Checkpoint nn_type ({checkpoint.get('nn_type')}) "
                             f"does not match current ({self.nn_type})")

        # Load from inference format into sparse training model
        self.model.load_from_inference_model(checkpoint['model_state_dict'])

        # Load optimizer states if available
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Could not load optimizer state, starting fresh")

        if self.optimizer_dense and checkpoint.get('optimizer_dense_state_dict'):
            try:
                self.optimizer_dense.load_state_dict(checkpoint['optimizer_dense_state_dict'])
            except:
                print("Warning: Could not load dense optimizer state, starting fresh")

        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("Warning: Could not load scheduler state, starting fresh")

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'lr': []})

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")

    def train(
            self,
            epochs: int,
            positions_per_epoch: int,
            checkpoint_path: str,
            early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train
            positions_per_epoch: Positions to process per epoch
            checkpoint_path: Path to save checkpoints
            early_stopping_patience: Epochs without improvement before stopping

        Returns:
            Training history dict
        """
        print(f"\n{'=' * 60}")
        print(f"Training {self.nn_type} Network (Memory Optimized)")
        print(f"{'=' * 60}")
        print(f"Train shards: {len(self.train_shards)}")
        print(f"Validation shards: {len(self.val_shards)}")
        print(f"Batch size: {self.batch_size:,}")
        print(f"Positions per epoch: {positions_per_epoch:,}")
        print(f"Device: {self.device}")
        print(f"{'=' * 60}\n")

        start_epoch = self.current_epoch + 1

        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Re-discover shards at start of each epoch to pick up new data
            prev_train = len(self.train_shards)
            prev_val = len(self.val_shards)
            total_shards = self._refresh_shards()
            new_train = len(self.train_shards)
            new_val = len(self.val_shards)

            if new_train != prev_train or new_val != prev_val:
                print(f"  Shards updated: {prev_train}+{prev_val} -> {new_train}+{new_val} (total: {total_shards})")

            print(f"Epoch {epoch}/{epochs}")
            print("-" * 40)

            # Train
            train_loss = self.train_epoch(positions_per_epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            print("  Validating...")
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Sync LR to sparse optimizer if using separate optimizers
            if self.optimizer_dense:
                new_lr = self.optimizer_dense.param_groups[0]['lr']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.1f}s")

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"  ✓ New best validation loss!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")

            # Save checkpoint
            if epoch % CHECKPOINT_INTERVAL == 0 or is_best:
                self.save_checkpoint(checkpoint_path, is_best)
                print(f"  Checkpoint saved to {checkpoint_path}")

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            print()

            # Force garbage collection between epochs
            gc.collect()

        # Final checkpoint
        self.save_checkpoint(checkpoint_path, False)

        return self.history


# Import discover_shards from shared module
from shard_io import discover_shards


def split_shards(shards: List[str], val_ratio: float = VALIDATION_SPLIT_RATIO) -> Tuple[List[str], List[str]]:
    """Split shards into training and validation sets."""
    num_val = max(1, int(len(shards) * val_ratio))
    num_train = len(shards) - num_val

    # Use last shards for validation (they're from later in processing)
    train_shards = shards[:num_train]
    val_shards = shards[num_train:]

    return train_shards, val_shards


def main():
    parser = argparse.ArgumentParser(
        description='Train NNUE or DNN model from binary shards (Memory Optimized).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train NNUE model
    python nn_train.py --nn-type NNUE --data-dir data/nnue

    # Train DNN model with custom parameters
    python nn_train.py --nn-type DNN --data-dir data/dnn --batch-size 8192 --lr 0.0005

    # Resume training from checkpoint
    python nn_train.py --nn-type NNUE --data-dir data/nnue --resume model/nnue.pt

Memory Optimization Notes:
    - Uses NumPy arrays instead of Python dicts (5-7x memory reduction)
    - Streaming decompression to avoid double memory usage
    - Default workers reduced to 2 (use --num-workers to adjust)
    - Buffered positions limited to 500K (configurable in code)
"""
    )

    parser.add_argument(
        '--nn-type',
        type=str,
        required=True,
        choices=['NNUE', 'DNN'],
        help='Neural network type'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing shard files (or parent directory with nnue/dnn subdirs)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to save checkpoints (default: model/{nn_type}.pt)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of epochs (default: {EPOCHS})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )

    parser.add_argument(
        '--positions-per-epoch',
        type=int,
        default=POSITIONS_PER_EPOCH,
        help=f'Positions per epoch (default: {POSITIONS_PER_EPOCH:,})'
    )

    parser.add_argument(
        '--val-size',
        type=int,
        default=VALIDATION_SIZE,
        help=f'Validation set size (default: {VALIDATION_SIZE:,})'
    )

    parser.add_argument(
        '--early-stopping',
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f'Early stopping patience (default: {EARLY_STOPPING_PATIENCE})'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,  # Reduced default from 4
        help='Number of parallel data loading workers (default: 2)'
    )

    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint_path = args.checkpoint or f"model/{args.nn_type.lower()}.pt"

    # Discover shards (Trainer will re-discover each epoch)
    print(f"Discovering shards in {args.data_dir}...")
    shards = discover_shards(args.data_dir, args.nn_type)

    if not shards:
        print(f"Error: No shard files found in {args.data_dir}")
        print("Expected files matching: *.bin.zst")
        sys.exit(1)

    print(f"Found {len(shards)} shard files")

    # Preview train/val split (actual split done in Trainer and refreshed each epoch)
    train_shards, val_shards = split_shards(shards, val_ratio=VALIDATION_SPLIT_RATIO)
    print(f"Train shards: {len(train_shards)}")
    print(f"Validation shards: {len(val_shards)}")

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data loading workers: {args.num_workers}")

    # Print embedding mode
    if EMBEDDING_BAG:
        print(f"Embedding mode: EmbeddingBag (sparse)")
    else:
        print(f"Embedding mode: Dense one-hot vectors")

    # Adjust learning rate for dense mode if not explicitly set
    lr = args.lr
    if not EMBEDDING_BAG and args.lr == LEARNING_RATE:
        lr = LEARNING_RATE_DENSE
        print(f"Adjusted learning rate for dense mode: {lr}")

    # Create training model based on embedding mode
    if args.nn_type == "NNUE":
        if EMBEDDING_BAG:
            model = NNUENetworkSparse(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
        else:
            model = NNUENetworkDense(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
    else:
        if EMBEDDING_BAG:
            model = DNNNetworkSparse(DNN_INPUT_SIZE)
        else:
            model = DNNNetworkDense(DNN_INPUT_SIZE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        nn_type=args.nn_type,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        lr=lr,
        num_workers=args.num_workers,
        use_embedding_bag=EMBEDDING_BAG,
        val_ratio=VALIDATION_SPLIT_RATIO
    )
    # Load checkpoint if resuming
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train(
        epochs=args.epochs,
        positions_per_epoch=args.positions_per_epoch,
        checkpoint_path=checkpoint_path,
        early_stopping_patience=args.early_stopping
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"\nNote: Checkpoint is saved in inference-compatible format.")
    print(f"      Use with nn_inference.py NNUENetwork/DNNNetwork directly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, Exiting...\n")
        exit()
        # response = input("\nKeyboardInterrupt detected. Type 'exit' to quit, Enter to continue: ").strip()
        # if response.lower() == "exit":
        #    print("Exiting...\n")
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        exit()