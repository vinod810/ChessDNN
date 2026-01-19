#!/usr/bin/env python3
"""
nn_train.py - Train NNUE or DNN models from pre-processed binary shards.

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
"""

import argparse
import os
import sys
import io
import struct
import glob
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
VALIDATION_SPLIT_RATIO = 0.5 # FIXME
BATCH_SIZE = 16384
LEARNING_RATE = 0.001
POSITIONS_PER_EPOCH = 1_000_000 # FIXME 100_000_000  # 100M positions per epoch
VALIDATION_SIZE = int(POSITIONS_PER_EPOCH / 10) # 10_000_000  # 10M positions for validation (10% of epoch)
EPOCHS = 500
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3
SHARDS_PER_BUFFER = 5  # Number of shards to load and shuffle together
PREFETCH_BATCHES = 2  # Number of batches to prefetch
#NUM_DATA_WORKERS = 6  # Number of parallel data loading threads

# Checkpoint configuration
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs
VAL_INTERVAL = 1  # Validate every N epochs

# Garbage collection
GC_INTERVAL = 1000  # Run GC every N batches

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
# Data Loading
# =============================================================================

class ShardReader:
    """Reads positions from compressed binary shard files."""

    def __init__(self, nn_type: str):
        self.nn_type = nn_type.upper()
        self.dctx = zstd.ZstdDecompressor()

    def read_shard(self, shard_path: str) -> List[Dict[str, Any]]:
        """
        Read all positions from a shard file.

        Returns:
            List of position dicts with keys depending on nn_type:
            - DNN: 'score_cp', 'features'
            - NNUE: 'score_cp', 'stm', 'white_features', 'black_features'
        """
        positions = []

        with open(shard_path, 'rb') as f:
            compressed_data = f.read()

        data = self.dctx.decompress(compressed_data)
        buf = io.BytesIO(data)

        while True:
            # Read score (int16)
            score_bytes = buf.read(2)
            if len(score_bytes) < 2:
                break
            score_cp = struct.unpack('<h', score_bytes)[0]

            if self.nn_type == "DNN":
                # Read num_features (uint8)
                num_features = struct.unpack('<B', buf.read(1))[0]
                # Read features (uint16[])
                features = []
                for _ in range(num_features):
                    features.append(struct.unpack('<H', buf.read(2))[0])

                positions.append({
                    'score_cp': score_cp,
                    'features': features
                })
            else:  # NNUE
                # Read stm (uint8)
                stm = struct.unpack('<B', buf.read(1))[0]
                # Read white features
                num_white = struct.unpack('<B', buf.read(1))[0]
                white_features = []
                for _ in range(num_white):
                    white_features.append(struct.unpack('<H', buf.read(2))[0])
                # Read black features
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


def create_dnn_batch_sparse(positions: List[Dict], device: torch.device):
    """
    Create sparse batch tensors for DNN training.

    Args:
        positions: List of position dicts with 'score_cp' and 'features'
        device: Target device

    Returns:
        (indices, offsets, targets)
        - indices: 1D tensor of all feature indices concatenated
        - offsets: 1D tensor marking start of each sample's features
        - targets: (batch_size, 1) tensor of tanh targets
    """
    #batch_size = len(positions)

    # Collect all indices and compute offsets
    all_indices = []
    offsets = [0]
    targets = []

    for pos in positions:
        features = pos['features']
        all_indices.extend(features)
        offsets.append(len(all_indices))
        targets.append(np.tanh(pos['score_cp'] / TANH_SCALE))

    # Remove last offset (not needed for EmbeddingBag)
    offsets = offsets[:-1]

    # Convert to tensors
    indices = torch.tensor(all_indices, dtype=torch.long, device=device)
    offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)

    return indices, offsets, targets


def create_nnue_batch_sparse(positions: List[Dict], device: torch.device):
    """
    Create sparse batch tensors for NNUE training.

    Args:
        positions: List of position dicts with 'score_cp', 'stm', 'white_features', 'black_features'
        device: Target device

    Returns:
        (white_indices, white_offsets, black_indices, black_offsets, stm, targets)
    """
    batch_size = len(positions)

    # Collect all indices and compute offsets
    white_indices = []
    white_offsets = [0]
    black_indices = []
    black_offsets = [0]
    stm_list = []
    targets = []

    for pos in positions:
        white_indices.extend(pos['white_features'])
        white_offsets.append(len(white_indices))

        black_indices.extend(pos['black_features'])
        black_offsets.append(len(black_indices))

        stm_list.append(float(pos['stm']))
        targets.append(np.tanh(pos['score_cp'] / TANH_SCALE))

    # Remove last offsets
    white_offsets = white_offsets[:-1]
    black_offsets = black_offsets[:-1]

    # Convert to tensors
    white_indices = torch.tensor(white_indices, dtype=torch.long, device=device)
    white_offsets = torch.tensor(white_offsets, dtype=torch.long, device=device)
    black_indices = torch.tensor(black_indices, dtype=torch.long, device=device)
    black_offsets = torch.tensor(black_offsets, dtype=torch.long, device=device)
    stm = torch.tensor(stm_list, dtype=torch.float32, device=device).unsqueeze(1)
    targets = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)

    return white_indices, white_offsets, black_indices, black_offsets, stm, targets


class DataLoader:
    """
    Efficient data loader with parallel shard reading and batch preparation.

    Architecture:
    - Multiple reader threads load shards in parallel into a shared buffer
    - One batcher thread creates batches from the buffer and puts them in the output queue
    - Main thread consumes batches from the queue while workers continue loading

    This ensures training is never blocked waiting for I/O.
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
            num_workers: int = 4
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

        # Queues for inter-thread communication
        self.shard_queue = queue.Queue()  # Shards to be read
        self.position_queue = queue.Queue(maxsize=shards_per_buffer)  # Loaded positions
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
                    positions = reader.read_shard(shard_path)
                    self.position_queue.put(positions, timeout=5.0)
                except Exception as e:
                    print(f"Warning: Worker {worker_id} error reading {shard_path}: {e}")

            except queue.Empty:
                continue

        # Signal batcher that this reader is done
        try:
            self.position_queue.put(None, timeout=5.0)
        except queue.Full:
            pass  # Batcher may have already stopped

    def _batcher_worker(self):
        """Worker thread that creates batches from loaded positions."""
        buffer = []
        shards_loaded = 0
        active_readers = self.num_workers

        while not self.stop_event.is_set():
            # Check if we've reached max positions
            with self._positions_yielded_lock:
                if self.max_positions and self.positions_yielded >= self.max_positions:
                    break

            # Try to get more positions from readers
            try:
                positions = self.position_queue.get(timeout=0.1)
                if positions is None:  # Reader finished
                    active_readers -= 1
                    if active_readers <= 0 and len(buffer) == 0:
                        break
                    continue

                buffer.extend(positions)
                shards_loaded += 1

                # Shuffle when we have enough data
                if shards_loaded >= self.shards_per_buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    shards_loaded = 0

            except queue.Empty:
                # No new positions, but we might have buffered data
                if active_readers <= 0 and len(buffer) < self.batch_size:
                    # All readers done and not enough data for another batch
                    break

            # Create batches while we have enough data
            while len(buffer) >= self.batch_size:
                # Check max positions again
                with self._positions_yielded_lock:
                    if self.max_positions and self.positions_yielded >= self.max_positions:
                        break

                batch_positions = buffer[:self.batch_size]
                buffer = buffer[self.batch_size:]

                try:
                    if self.nn_type == "DNN":
                        batch = create_dnn_batch_sparse(batch_positions, self.device)
                    else:
                        batch = create_nnue_batch_sparse(batch_positions, self.device)

                    self.batch_queue.put(batch, timeout=5.0)

                    with self._positions_yielded_lock:
                        self.positions_yielded += len(batch_positions)

                except queue.Full:
                    # Queue full, put positions back
                    buffer = batch_positions + buffer
                    break
                except Exception as e:
                    print(f"Warning: Error creating batch: {e}")

        # Process remaining positions in buffer
        while len(buffer) >= self.batch_size:
            with self._positions_yielded_lock:
                if self.max_positions and self.positions_yielded >= self.max_positions:
                    break

            batch_positions = buffer[:self.batch_size]
            buffer = buffer[self.batch_size:]

            try:
                if self.nn_type == "DNN":
                    batch = create_dnn_batch_sparse(batch_positions, self.device)
                else:
                    batch = create_nnue_batch_sparse(batch_positions, self.device)

                self.batch_queue.put(batch, timeout=5.0)

                with self._positions_yielded_lock:
                    self.positions_yielded += len(batch_positions)
            except:
                break

        # Signal end
        self.batch_queue.put(None)

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
        try:
            while True:
                self.shard_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self.position_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self.batch_queue.get_nowait()
        except queue.Empty:
            pass

        # Wait for threads
        for t in self.reader_threads:
            t.join(timeout=2.0)
        if self.batcher_thread:
            self.batcher_thread.join(timeout=2.0)

    def __iter__(self):
        """Iterate over batches."""
        self.start()
        try:
            while True:
                try:
                    batch = self.batch_queue.get(timeout=30.0)
                    if batch is None:
                        break
                    yield batch
                except queue.Empty:
                    # Check if batcher is still running
                    if self.batcher_thread and not self.batcher_thread.is_alive():
                        break
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
            train_shards: List[str],
            val_shards: List[str],
            device: torch.device,
            batch_size: int = BATCH_SIZE,
            lr: float = LEARNING_RATE,
            num_workers: int = 4
    ):
        self.model = model.to(device)
        self.nn_type = nn_type.upper()
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

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

    def _forward_dnn(self, batch):
        """Forward pass for DNN with sparse batch."""
        indices, offsets, targets = batch
        outputs = self.model(indices, offsets)
        return outputs, targets

    def _forward_nnue(self, batch):
        """Forward pass for NNUE with sparse batch."""
        white_indices, white_offsets, black_indices, black_offsets, stm, targets = batch
        outputs = self.model(white_indices, white_offsets, black_indices, black_offsets, stm)
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
            num_workers=self.num_workers
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

            # Progress reporting
            #if num_batches % 100 == 0:
            #    elapsed = time.time() - start_time
            #    positions = num_batches * self.batch_size
            #    pos_per_sec = positions / elapsed if elapsed > 0 else 0
            #    print(f"\r  Batch {num_batches} | Loss: {loss.item():.6f} | "
            #          f"Positions: {positions:,} | {pos_per_sec:,.0f} pos/sec", end='')
            # Progress reporting (~10 times per epoch)
            if num_batches % report_interval == 0:
                pct = 100 * num_batches * self.batch_size / positions_per_epoch
                print(f"\r  [{pct:5.1f}%] Loss: {loss.item():.6f}", end='', flush=True)

            # Garbage collection
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
            num_workers=self.num_workers
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
        print(f"Training {self.nn_type} Network")
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

        # Final checkpoint
        self.save_checkpoint(checkpoint_path, False)

        return self.history


def discover_shards(data_dir: str, nn_type: str) -> List[str]:
    """Discover all shard files in a directory."""
    pattern = os.path.join(data_dir, "*.bin.zst")
    shards = sorted(glob.glob(pattern))

    if not shards:
        # Try looking in nn_type subdirectory
        pattern = os.path.join(data_dir, nn_type.lower(), "*.bin.zst")
        shards = sorted(glob.glob(pattern))

    return shards


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
        description='Train NNUE or DNN model from binary shards.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train NNUE model
    python nn_train.py --nn-type NNUE --data-dir data/nnue

    # Train DNN model with custom parameters
    python nn_train.py --nn-type DNN --data-dir data/dnn --batch-size 8192 --lr 0.0005

    # Resume training from checkpoint
    python nn_train.py --nn-type NNUE --data-dir data/nnue --resume model/nnue.pt
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
        default=4,
        help='Number of parallel data loading workers (default: 4)'
    )

    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint_path = args.checkpoint or f"model/{args.nn_type.lower()}.pt"

    # Discover shards
    print(f"Discovering shards in {args.data_dir}...")
    shards = discover_shards(args.data_dir, args.nn_type)

    if not shards:
        print(f"Error: No shard files found in {args.data_dir}")
        print("Expected files matching: *.bin.zst")
        sys.exit(1)

    print(f"Found {len(shards)} shard files")

    # Split into train/val
    train_shards, val_shards = split_shards(shards, val_ratio=VALIDATION_SPLIT_RATIO)
    print(f"Train shards: {len(train_shards)}")
    print(f"Validation shards: {len(val_shards)}")

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data loading workers: {args.num_workers}")

    # Create sparse training model
    if args.nn_type == "NNUE":
        model = NNUENetworkSparse(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
    else:
        model = DNNNetworkSparse(DNN_INPUT_SIZE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        nn_type=args.nn_type,
        train_shards=train_shards,
        val_shards=val_shards,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
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
        response = input("\nKeyboardInterrupt detected. Type 'exit' to quit, Enter to continue: ").strip()
        if response.lower() == "exit":
            print("Resuming...\n")
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
