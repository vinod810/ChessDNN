import argparse
import numpy as np
import chess.pgn
import zstandard as zstd
import io
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any
import random
import os
import platform
import gc
import time
import glob
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from collections import deque
import threading
import ctypes
import chess
from typing import List, Tuple, Dict, Optional

from nn_inference import NNUENetwork, DNNNetwork, \
    NNUE_INPUT_SIZE, DNN_INPUT_SIZE, NNUE_HIDDEN_SIZE, \
    DNN_HIDDEN_LAYERS, NNUEFeatures, MAX_SCORE, NNUEIncrementalUpdater, \
    TANH_SCALE, DNNFeatures

MATE_FACTOR = 100
MAX_MATE_DEPTH = 10
MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR

"""
Chess Neural Network Training Script
Supports both NNUE and DNN architectures for chess position evaluation.

ARCHITECTURE SELECTION:
    Set nn_type = "NNUE" or nn_type = "DNN" in the Configuration section below.

NNUE (Efficiently Updatable Neural Network):
    - Input: Two 40960-dimensional sparse vectors (white/black king-piece features)
    - Architecture: 40960 -> 256 (shared) -> 512 (concatenated) -> 32 -> 32 -> 1
    - Hidden activation: Clipped ReLU [0, 1]
    - Output activation: Linear (no activation)
    - Features: King-relative piece positions for both perspectives
    - Output: Position evaluation from side-to-move's perspective

DNN (Deep Neural Network):
    - Input: Single 768-dimensional one-hot encoded vector (from player's perspective)
    - Architecture: 768 -> 1024 -> 256 -> 32 -> 1
    - Hidden activation: Clipped ReLU [0, 1]
    - Output activation: Linear (no activation)
    - Features: Piece positions from perspective of player to move
    - Output: Position evaluation from side-to-move's perspective

Both networks are trained on tanh-normalized targets: tanh(centipawns / 400).
Both networks output linearly and learn to produce values in approximately [-1, 1].
Output range: approximately [-1, 1] for both architectures.
"""

# Configuration
# Network type selection: "NNUE" or "DNN"
NN_TYPE = "NNUE"
INPUT_SIZE = NNUE_INPUT_SIZE if NN_TYPE == "NNUE" else DNN_INPUT_SIZE
FIRST_HIDDEN_SIZE = NNUE_HIDDEN_SIZE if NN_TYPE == "NNUE" else DNN_HIDDEN_LAYERS[0]
MODEL_PATH = "model/nnue.pt" if NN_TYPE == "NNUE" else "model/dnn.pt"

# NNUE Configuration

# DNN Configuration

# Dynamic configuration based on nn_type

# Main configuration
BATCH_SIZE = 8192 * 2
if platform.system() == "Windows":
    QUEUE_READ_TIMEOUT = int(BATCH_SIZE / 512) * 2  # Windows file reading is slow
else:
    QUEUE_READ_TIMEOUT = int(BATCH_SIZE / 512)

# Worker configuration
QUEUE_MAX_SIZE = 100  # Max batches in queue
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 10

# Misc
GC_INTERVAL = 1000  # Run garbage collection every N batches

# Training
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.05
STEPS_PER_EPOCH = int(8 * 1024 * 1024 / BATCH_SIZE)
POSITIONS_PER_EPOCH = BATCH_SIZE * STEPS_PER_EPOCH
EPOCHS = 500
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3

# PGN
MAX_PLYS_PER_GAME = 200
OPENING_PLYS = 10


class SharedStats:
    """Thread-safe shared statistics using multiprocessing Values"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers

        # Worker stats (arrays indexed by worker_id)
        self.worker_games = [Value(ctypes.c_uint64, 0) for _ in range(num_workers)]
        self.worker_positions = [Value(ctypes.c_uint64, 0) for _ in range(num_workers)]
        self.worker_batches = [Value(ctypes.c_uint64, 0) for _ in range(num_workers)]
        self.worker_file_loops = [Value(ctypes.c_uint64, 0) for _ in range(num_workers)]
        self.worker_wait_ms = [Value(ctypes.c_uint64, 0) for _ in range(num_workers)]
        self.worker_process_ms = [Value(ctypes.c_uint64, 0) for _ in range(num_workers)]

        # Main process stats
        self.main_batches = Value(ctypes.c_uint64, 0)
        self.main_train_batches = Value(ctypes.c_uint64, 0)
        self.main_val_batches = Value(ctypes.c_uint64, 0)
        self.main_wait_ms = Value(ctypes.c_uint64, 0)
        self.main_process_ms = Value(ctypes.c_uint64, 0)

        # Queue stats
        self.queue_full_count = Value(ctypes.c_uint64, 0)
        self.queue_empty_count = Value(ctypes.c_uint64, 0)

    def get_worker_stats(self, worker_id: int) -> Dict[str, Any]:
        return {
            'games': self.worker_games[worker_id].value,
            'positions': self.worker_positions[worker_id].value,
            'batches': self.worker_batches[worker_id].value,
            'file_loops': self.worker_file_loops[worker_id].value,
            'wait_seconds': self.worker_wait_ms[worker_id].value / 1000.0,
            'process_seconds': self.worker_process_ms[worker_id].value / 1000.0,
        }

    def get_main_stats(self) -> Dict[str, Any]:
        return {
            'batches': self.main_batches.value,
            'train_batches': self.main_train_batches.value,
            'val_batches': self.main_val_batches.value,
            'wait_seconds': self.main_wait_ms.value / 1000.0,
            'process_seconds': self.main_process_ms.value / 1000.0,
        }

    def print_stats(self, file_paths: List[str]):
        """Print formatted statistics"""
        print("\n" + "=" * 80)
        print("PERFORMANCE STATISTICS")
        print("=" * 80)

        total_worker_wait = 0
        total_worker_process = 0

        for i in range(self.num_workers):
            stats = self.get_worker_stats(i)
            total_worker_wait += stats['wait_seconds']
            total_worker_process += stats['process_seconds']

            file_name = os.path.basename(file_paths[i]) if i < len(file_paths) else f"worker_{i}"
            print(f"\nWorker {i} ({file_name}):")
            print(f"  Games: {stats['games']:,} | Positions: {stats['positions']:,} | "
                  f"Batches: {stats['batches']:,}")
            print(f"  File loops: {stats['file_loops']} | "
                  f"Wait: {stats['wait_seconds']:.1f}s | Process: {stats['process_seconds']:.1f}s")

            if stats['wait_seconds'] + stats['process_seconds'] > 0:
                wait_pct = stats['wait_seconds'] / (stats['wait_seconds'] + stats['process_seconds']) * 100
                print(f"  Wait ratio: {wait_pct:.1f}%")

        main_stats = self.get_main_stats()
        print(f"\nMain Process:")
        print(f"  Batches consumed: {main_stats['batches']:,} "
              f"(Train: {main_stats['train_batches']:,}, Val: {main_stats['val_batches']:,})")
        print(f"  Wait: {main_stats['wait_seconds']:.1f}s | Process: {main_stats['process_seconds']:.1f}s")

        if main_stats['wait_seconds'] + main_stats['process_seconds'] > 0:
            wait_pct = main_stats['wait_seconds'] / (main_stats['wait_seconds'] + main_stats['process_seconds']) * 100
            print(f"  Wait ratio: {wait_pct:.1f}%")

        print(f"\nQueue Events:")
        print(f"  Queue full events: {self.queue_full_count.value:,}")
        print(f"  Queue empty events: {self.queue_empty_count.value:,}")

        # Analysis
        print(f"\nANALYSIS:")
        avg_worker_wait = total_worker_wait / max(1, self.num_workers)
        if avg_worker_wait > main_stats['wait_seconds'] * 1.5:
            print("  ⚠ Workers waiting more than main - queue may be full often")
            print("    Consider: increase QUEUE_MAX_SIZE or reduce worker count")
        elif main_stats['wait_seconds'] > avg_worker_wait * 1.5:
            print("  ⚠ Main process waiting more than workers - queue often empty")
            print("    Consider: add more workers or increase batch size")
        else:
            print("  ✓ Balanced: workers and main process have similar wait times")

        print("=" * 80)


def encode_sparse_batch(positions: List[Tuple]) -> Dict[str, Any]:
    """
    OPTIMIZED: Encode with flattened numpy arrays for faster IPC.

    Uses offset arrays to reconstruct per-sample indices, which pickles
    much faster than lists of lists.
    """
    if NN_TYPE == "NNUE":
        all_white_indices = []
        all_black_indices = []
        white_offsets = [0]
        black_offsets = [0]
        stms = []
        scores = []

        for white_feat, black_feat, stm, score in positions:
            all_white_indices.extend(white_feat)
            all_black_indices.extend(black_feat)
            white_offsets.append(len(all_white_indices))
            black_offsets.append(len(all_black_indices))
            stms.append(stm)
            scores.append(score)

        return {
            'white_indices_flat': np.array(all_white_indices, dtype=np.int32),
            'white_offsets': np.array(white_offsets, dtype=np.int32),
            'black_indices_flat': np.array(all_black_indices, dtype=np.int32),
            'black_offsets': np.array(black_offsets, dtype=np.int32),
            'stm': np.array(stms, dtype=np.float32),
            'scores': np.array(scores, dtype=np.float32),
            'batch_size': len(positions),
            'format': 'flattened'  # Flag to identify format
        }
    else:  # DNN
        all_features = []
        offsets = [0]
        scores = []

        for feat, score in positions:
            all_features.extend(feat)
            offsets.append(len(all_features))
            scores.append(score)

        return {
            'features_flat': np.array(all_features, dtype=np.int32),
            'offsets': np.array(offsets, dtype=np.int32),
            'scores': np.array(scores, dtype=np.float32),
            'batch_size': len(positions),
            'format': 'flattened'
        }


def decode_sparse_batch(batch_data: Dict[str, Any], device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
    """
    OPTIMIZED: Decode flattened numpy format using vectorized operations.

    Uses np.repeat to build batch indices from offsets - fully vectorized,
    no Python loops over batch_size.
    """
    batch_size = batch_data['batch_size']
    scores = torch.tensor(batch_data['scores'], dtype=torch.float32, device=device).unsqueeze(1)

    if NN_TYPE == "NNUE":
        white_input = torch.zeros(batch_size, INPUT_SIZE, device=device)
        black_input = torch.zeros(batch_size, INPUT_SIZE, device=device)

        # Get flattened arrays and offsets
        white_flat = batch_data['white_indices_flat']
        white_offsets = batch_data['white_offsets']
        black_flat = batch_data['black_indices_flat']
        black_offsets = batch_data['black_offsets']

        # Build batch indices from offsets using np.repeat (fully vectorized)
        # np.diff gives counts per sample, np.repeat expands to batch indices
        if len(white_flat) > 0:
            white_batch_idx = np.repeat(np.arange(batch_size, dtype=np.int64), np.diff(white_offsets))
            white_input[white_batch_idx, white_flat] = 1.0

        if len(black_flat) > 0:
            black_batch_idx = np.repeat(np.arange(batch_size, dtype=np.int64), np.diff(black_offsets))
            black_input[black_batch_idx, black_flat] = 1.0

        stm = torch.tensor(batch_data['stm'], dtype=torch.float32, device=device).unsqueeze(1)
        return white_input, black_input, stm, scores

    else:  # DNN
        features = torch.zeros(batch_size, INPUT_SIZE, device=device)

        feat_flat = batch_data['features_flat']
        offsets = batch_data['offsets']

        if len(feat_flat) > 0:
            batch_idx = np.repeat(np.arange(batch_size, dtype=np.int64), np.diff(offsets))
            features[batch_idx, feat_flat] = 1.0

        return features, scores


def worker_process(
        worker_id: int,
        pgn_file: str,
        output_queue: Queue,
        stop_event: Event,
        stats: SharedStats,
        batch_size: int = BATCH_SIZE,
        max_positions_per_game: int = MAX_PLYS_PER_GAME,
        opening_plys: int = OPENING_PLYS,
        shuffle_buffer_size: int = SHUFFLE_BUFFER_SIZE,  # 2000  # Reduced from 10000 for faster startup
        skip_games_count: int = 0  # Number of games to skip on first file loop (for resume)
):
    """
    Worker process that streams positions from a PGN file.
    Loops through the file repeatedly until stop_event is set.
    Uses a shuffle buffer to randomize position order.

    Args:
        skip_games_count: Number of games to skip at the start of the first file loop.
                          Used when resuming training to avoid reprocessing games.
    """
    print(f"Worker {worker_id} starting: {os.path.basename(pgn_file)}" +
          (f" (skipping first {skip_games_count} games)" if skip_games_count > 0 else ""))

    position_buffer = []
    gc_counter = 0
    is_first_file_loop = True  # Track if this is the first pass through the file
    games_skipped = 0  # Counter for skipped games

    # Create local game processor with validation (no shared state)
    game_processor = ProcessGameWithValidation(NN_TYPE)

    while not stop_event.is_set():
        try:
            # Open and stream from file
            with open(pgn_file, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                    while not stop_event.is_set():
                        process_start = time.time()

                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            # EOF reached, increment loop counter and break to restart
                            # with stats.worker_file_loops[worker_id].get_lock(): # Rare event so lock is fine
                            stats.worker_file_loops[worker_id].value += 1
                            is_first_file_loop = False  # No longer first loop after completing file
                            break

                        # Skip games on the first file loop if resuming
                        if is_first_file_loop and games_skipped < skip_games_count:
                            games_skipped += 1
                            if games_skipped % 10000 == 0:
                                print(f"Worker {worker_id}: Skipped {games_skipped}/{skip_games_count} games...")
                            continue

                        # Process game with periodic validation (using local counter)
                        positions = game_processor(game, max_positions_per_game, opening_plys)

                        if positions:
                            position_buffer.extend(positions)
                            # with stats.worker_games[worker_id].get_lock():
                            stats.worker_games[worker_id].value += 1
                            # with stats.worker_positions[worker_id].get_lock():
                            stats.worker_positions[worker_id].value += len(positions)

                        process_time = time.time() - process_start
                        # with stats.worker_process_ms[worker_id].get_lock():
                        stats.worker_process_ms[worker_id].value += int(process_time * 1000)

                        # When buffer is large enough, shuffle and send batches
                        while len(position_buffer) >= shuffle_buffer_size and not stop_event.is_set():
                            # Shuffle the buffer
                            random.shuffle(position_buffer)

                            # Send batches until buffer is half empty
                            while len(
                                    position_buffer) >= shuffle_buffer_size // 2 + batch_size and not stop_event.is_set():
                                batch_positions = position_buffer[:batch_size]
                                position_buffer = position_buffer[batch_size:]

                                # Encode to sparse format
                                sparse_batch = encode_sparse_batch(batch_positions)

                                # Try to put in queue, track wait time
                                wait_start = time.time()
                                while not stop_event.is_set():
                                    try:
                                        output_queue.put(sparse_batch, timeout=0.1)
                                        # with stats.worker_batches[worker_id].get_lock():
                                        stats.worker_batches[worker_id].value += 1
                                        break
                                    except:
                                        # Queue full
                                        # with stats.queue_full_count.get_lock():
                                        stats.queue_full_count.value += 1

                                wait_time = time.time() - wait_start
                                # with stats.worker_wait_ms[worker_id].get_lock():
                                stats.worker_wait_ms[worker_id].value += int(wait_time * 1000)

                                # Clear batch_positions explicitly
                                del batch_positions
                                del sparse_batch

                        # Periodic garbage collection
                        gc_counter += 1
                        if gc_counter >= GC_INTERVAL:
                            gc.collect()
                            gc_counter = 0

                    # Clean up text stream
                    del text_stream

            # File loop completed, garbage collect
            gc.collect()

        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # Brief pause before retry

    # Send remaining positions (shuffle first)
    if position_buffer and not stop_event.is_set():
        random.shuffle(position_buffer)
        while len(position_buffer) >= batch_size:
            batch_positions = position_buffer[:batch_size]
            position_buffer = position_buffer[batch_size:]
            sparse_batch = encode_sparse_batch(batch_positions)
            try:
                output_queue.put(sparse_batch, timeout=1.0)
            except:
                break
        # Send any remaining as partial batch
        if position_buffer:
            sparse_batch = encode_sparse_batch(position_buffer)
            try:
                output_queue.put(sparse_batch, timeout=1.0)
            except:
                pass

    # Clear buffer
    position_buffer.clear()
    gc.collect()

    print(f"Worker {worker_id} stopped")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, min_delta: float = 0.0, verbose: bool = True,
                 checkpoint_path: str = MODEL_PATH, initial_best_loss: float = None, initial_best_epoch: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = initial_best_loss
        self.early_stop = False
        self.best_epoch = initial_best_epoch
        # Store references for checkpoint saving
        self.optimizer = None
        self.scheduler = None

    def set_optimizer_scheduler(self, optimizer, scheduler):
        """Set optimizer and scheduler references for checkpoint saving"""
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self._save_checkpoint(model, epoch, val_loss)
            if self.verbose:
                print(f"  First validation loss: {val_loss:.6f}")
                print(f"  Saved checkpoint to {self.checkpoint_path}")
            return False

        if val_loss < self.best_loss - self.min_delta:
            improvement = self.best_loss - val_loss
            if self.verbose:
                print(f"  Validation loss improved by {improvement:.6f}")
            self.best_loss = val_loss
            self.best_epoch = epoch
            self._save_checkpoint(model, epoch, val_loss)
            if self.verbose:
                print(f"  Saved checkpoint to {self.checkpoint_path}")
            self.counter = 0
            return False

        self.counter += 1
        if self.verbose:
            print(f"  No improvement for {self.counter}/{self.patience} epochs")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"  Early stopping triggered! Best loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
            return True

        return False

    def _save_checkpoint(self, model: nn.Module, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'nn_type': NN_TYPE,
        }
        # Save optimizer state if available
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        # Save scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_path)

    def restore_best_model(self, model: nn.Module):
        if self.checkpoint_path and self.best_loss is not None:
            checkpoint = torch.load(self.checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Restored model from epoch {checkpoint['epoch']} "
                  f"with validation loss: {checkpoint['val_loss']:.6f}")


class ParallelTrainer:
    """Main training coordinator with parallel data loading"""

    def __init__(
            self,
            pgn_dir: str,
            model: nn.Module,
            batch_size: int = BATCH_SIZE,
            validation_split: float = VALIDATION_SPLIT,
            queue_size: int = QUEUE_MAX_SIZE,
            device: str = 'cpu',
            seed: int = 42,
            skip_games_count: int = 0  # Games to skip per file when resuming
    ):
        self.pgn_dir = pgn_dir
        self.model = model.to(device)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.queue_size = queue_size
        self.device = device
        self.seed = seed
        self.skip_games_count = skip_games_count

        # Find all PGN files
        self.pgn_files = sorted(glob.glob(os.path.join(pgn_dir, "*.pgn.zst")))
        if not self.pgn_files:
            raise ValueError(f"No .pgn.zst files found in {pgn_dir}")

        print(f"Found {len(self.pgn_files)} PGN files:")
        for f in self.pgn_files:
            print(f"  - {os.path.basename(f)}")

        self.num_workers = len(self.pgn_files)

        # Multiprocessing components
        self.data_queue = None
        self.stop_event = None
        self.workers = []
        self.stats = None

        # Training state
        self.rng = random.Random(seed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)  # Use self.model, not model
        self.criterion = nn.MSELoss()

        # Validation buffer with size limit
        self.val_buffer = deque(maxlen=1000)  # Limit validation buffer size in batches
        self.val_buffer_lock = threading.Lock()

    def start_workers(self):
        """Start all worker processes"""
        self.data_queue = mp.Queue(maxsize=self.queue_size)
        self.stop_event = mp.Event()
        self.stats = SharedStats(self.num_workers)

        for i, pgn_file in enumerate(self.pgn_files):
            p = Process(
                target=worker_process,
                args=(i, pgn_file, self.data_queue, self.stop_event, self.stats,
                      self.batch_size),
                kwargs={'skip_games_count': self.skip_games_count}
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

        print(f"Started {len(self.workers)} worker processes" +
              (f" (skipping first {self.skip_games_count} games per file)"
               if self.skip_games_count > 0 else ""))

    def stop_workers(self):
        """Stop all worker processes and clean up"""
        print("Stopping workers...")
        self.stop_event.set()

        # Drain the queue to unblock workers
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        self.workers.clear()

        # Clean up queue
        self.data_queue.close()
        self.data_queue.join_thread()

        print("All workers stopped")

    def get_batch(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a batch from the queue with timeout"""
        wait_start = time.time()

        while True:
            try:
                batch = self.data_queue.get(timeout=timeout)
                wait_time = time.time() - wait_start
                # with self.stats.main_wait_ms.get_lock():
                self.stats.main_wait_ms.value += int(wait_time * 1000)
                return batch
            except:
                # with self.stats.queue_empty_count.get_lock():
                self.stats.queue_empty_count.value += 1

                # Check if all workers are dead
                alive_workers = sum(1 for p in self.workers if p.is_alive())
                if alive_workers == 0:
                    return None

                if time.time() - wait_start > timeout:
                    return None

    def train_epoch(self, positions_per_epoch: int = POSITIONS_PER_EPOCH) -> Tuple[float, float]:
        """
        Train for one epoch.
        Returns (train_loss, val_loss)
        """
        self.model.train()

        total_train_loss = 0
        train_batch_count = 0
        positions_processed = 0
        gc_counter = 0

        # Clear old validation data
        with self.val_buffer_lock:
            self.val_buffer.clear()

        while positions_processed < positions_per_epoch:
            batch_data = self.get_batch(timeout=QUEUE_READ_TIMEOUT)

            if batch_data is None:
                print("Warning: No batch received, waiting...")
                continue

            process_start = time.time()

            # Train/validation split (main process decides)
            # is_validation = self.rng.random() < self.validation_split
            is_validation = self.stats.main_val_batches.value < positions_per_epoch / BATCH_SIZE * self.validation_split

            # with self.stats.main_batches.get_lock():
            self.stats.main_batches.value += 1

            if is_validation:
                # Store for validation (with size limit via deque maxlen)
                with self.val_buffer_lock:
                    self.val_buffer.append(batch_data)
                # with self.stats.main_val_batches.get_lock():
                self.stats.main_val_batches.value += 1
            else:
                # Training step
                if NN_TYPE == "NNUE":
                    white_input, black_input, stm, target = decode_sparse_batch(
                        batch_data, self.device
                    )
                    self.optimizer.zero_grad()
                    output = self.model(white_input, black_input, stm)
                    loss = self.criterion(output, target)
                    loss.backward()

                    # Gradient clipping for stability
                    if hasattr(self, 'grad_clip') and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()

                    total_train_loss += loss.item()
                    train_batch_count += 1
                    positions_processed += batch_data['batch_size']

                    # with self.stats.main_train_batches.get_lock():
                    self.stats.main_train_batches.value += 1

                    # Clean up tensors
                    del white_input, black_input, stm, target, output, loss

                else:  # DNN
                    features, target = decode_sparse_batch(
                        batch_data, self.device
                    )
                    self.optimizer.zero_grad()
                    output = self.model(features)
                    loss = self.criterion(output, target)
                    loss.backward()

                    # Gradient clipping for stability
                    if hasattr(self, 'grad_clip') and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()

                    total_train_loss += loss.item()
                    train_batch_count += 1
                    positions_processed += batch_data['batch_size']

                    # with self.stats.main_train_batches.get_lock():
                    self.stats.main_train_batches.value += 1

                    # Clean up tensors
                    del features, target, output, loss

            # Clean up batch data
            del batch_data

            process_time = time.time() - process_start
            # with self.stats.main_process_ms.get_lock():
            self.stats.main_process_ms.value += int(process_time * 1000)

            # Progress update
            if train_batch_count != 0 and train_batch_count % int(
                    POSITIONS_PER_EPOCH / BATCH_SIZE / 50) == 0:  # 50 prints per epoch
                avg_loss = total_train_loss / max(1, train_batch_count)
                print(f"  Batch {train_batch_count}: Loss={avg_loss:.6f}, "
                      f"Positions={positions_processed:,}/{positions_per_epoch:,}")

            # Periodic garbage collection
            gc_counter += 1
            if gc_counter >= GC_INTERVAL:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc_counter = 0

        # Calculate validation loss
        val_loss = self._compute_validation_loss()

        avg_train_loss = total_train_loss / max(1, train_batch_count)

        # Clear validation buffer after computing loss
        with self.val_buffer_lock:
            self.val_buffer.clear()
        # with self.stats.main_val_batches.get_lock():
        self.stats.main_val_batches.value = 0

        gc.collect()

        return avg_train_loss, val_loss

    def _compute_validation_loss(self) -> float:
        """Compute validation loss from buffered batches"""
        self.model.eval()
        total_loss = 0
        batch_count = 0

        with self.val_buffer_lock:
            val_batches = list(self.val_buffer)
            print(f"Computing validation loss, val_batches size={len(val_batches)}...")

        with torch.no_grad():
            for batch_data in val_batches:
                if NN_TYPE == "NNUE":
                    white_input, black_input, stm, target = decode_sparse_batch(
                        batch_data, self.device
                    )
                    output = self.model(white_input, black_input, stm)
                    loss = self.criterion(output, target)
                    total_loss += loss.item()
                    batch_count += 1
                    # Clean up
                    del white_input, black_input, stm, target, output, loss
                else:  # DNN
                    features, target = decode_sparse_batch(
                        batch_data, self.device
                    )
                    output = self.model(features)
                    loss = self.criterion(output, target)
                    total_loss += loss.item()
                    batch_count += 1
                    # Clean up
                    del features, target, output, loss

        # Clear the local copy
        del val_batches

        self.model.train()
        return total_loss / max(1, batch_count)

    def train(
            self,
            epochs: int = EPOCHS,
            lr: float = LEARNING_RATE,
            positions_per_epoch: int = POSITIONS_PER_EPOCH,
            early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
            checkpoint_path: str = MODEL_PATH,
            lr_scheduler: str = "plateau",  # "plateau", "step", or "none"
            grad_clip: float = 1.0,  # Gradient clipping max norm
            resume_checkpoint: dict = None  # Checkpoint dict for resuming
    ) -> Dict[str, List[float]]:
        """Main training loop with LR scheduling and gradient clipping

        Args:
            epochs: Total number of epochs to train
            lr: Learning rate (ignored if resuming with saved optimizer state)
            positions_per_epoch: Number of positions per epoch
            early_stopping_patience: Patience for early stopping
            checkpoint_path: Path to save checkpoints
            lr_scheduler: Type of learning rate scheduler
            grad_clip: Gradient clipping max norm
            resume_checkpoint: Optional checkpoint dict to resume from
        """
        start_epoch = 0
        initial_best_loss = None
        initial_best_epoch = 0

        # Handle resuming from checkpoint
        if resume_checkpoint is not None:
            start_epoch = resume_checkpoint.get('epoch', 0)
            initial_best_loss = resume_checkpoint.get('best_loss', resume_checkpoint.get('val_loss'))
            initial_best_epoch = resume_checkpoint.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch} with best loss {initial_best_loss:.6f}")

        # Update learning rate (only if not resuming with optimizer state)
        if resume_checkpoint is None or 'optimizer_state_dict' not in resume_checkpoint:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Setup learning rate scheduler
        scheduler = None
        if lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=LR_PATIENCE,
                min_lr=1e-6
            )
        elif lr_scheduler == "step":  # Not used
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.5
            )

        # Load optimizer and scheduler state if resuming
        if resume_checkpoint is not None:
            if 'optimizer_state_dict' in resume_checkpoint:
                self.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                print("Restored optimizer state from checkpoint")
            if 'scheduler_state_dict' in resume_checkpoint and scheduler is not None:
                scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
                print("Restored scheduler state from checkpoint")

        self.grad_clip = grad_clip

        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            checkpoint_path=checkpoint_path,
            initial_best_loss=initial_best_loss,
            initial_best_epoch=initial_best_epoch
        )
        # Provide optimizer and scheduler to early stopping for checkpoint saving
        early_stopping.set_optimizer_scheduler(self.optimizer, scheduler)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}

        self.start_workers()

        try:
            for epoch in range(start_epoch, epochs):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch + 1}/{epochs} (LR: {current_lr:.6f})")
                print('=' * 60)

                train_loss, val_loss = self.train_epoch(positions_per_epoch)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['lr'].append(current_lr)

                print(f"\n  Train Loss: {train_loss:.6f}")
                print(f"  Validation Loss: {val_loss:.6f}")

                # Update learning rate scheduler
                if scheduler is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    if lr_scheduler == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

                if early_stopping(val_loss, self.model, epoch + 1):
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                # Print stats periodically
                if (epoch + 1) % 5 == 0:
                    self.stats.print_stats(self.pgn_files)

        finally:
            self.stop_workers()
            early_stopping.restore_best_model(self.model)

            # Final stats
            self.stats.print_stats(self.pgn_files)

        return history


class ProcessGameWithValidation:
    """
    Callable class that processes games with periodic validation of incremental features.
    Each instance maintains its own position counter (no global state).
    """

    VALIDATION_INTERVAL = 10000

    def __init__(self, nn_type):
        self.position_count = 0
        self.nn_type = nn_type

    def is_matching_full_vs_incremental(self, board, white_feat, black_feat):
        # Full extraction for comparison
        white_feat_full, black_feat_full = NNUEFeatures.board_to_features(board)

        # Compare as sets (order doesn't matter)
        white_match = set(white_feat) == set(white_feat_full)
        black_match = set(black_feat) == set(black_feat_full)

        if not white_match or not black_match:
            print(f"\n⚠️  WARNING: Incremental feature mismatch at position {self.position_count}!")
            print(f"    FEN: {board.fen()}")
            if not white_match:
                incremental_only = set(white_feat) - set(white_feat_full)
                full_only = set(white_feat_full) - set(white_feat)
                print(f"    White features - Incremental only: {incremental_only}")
                print(f"    White features - Full only: {full_only}")
            if not black_match:
                incremental_only = set(black_feat) - set(black_feat_full)
                full_only = set(black_feat_full) - set(black_feat)
                print(f"    Black features - Incremental only: {incremental_only}")
                print(f"    Black features - Full only: {full_only}")

            return False
        else:
            return True

    @staticmethod
    def eval_to_cp_stm(ev, board_turn: bool) -> Optional[int]:
        """
        Convert a chess.engine evaluation to centipawns from side-to-move perspective.

        Args:
            ev: The evaluation from node.eval() (can be None)
            board_turn: True if white to move, False if black to move

        Returns:
            Centipawn score from STM perspective, or None if ev is None
        """
        if ev is None:
            return None

        if ev.is_mate():
            mate_in = ev.white().mate()
            if mate_in < 0:  # -ve when black is winning
                mate_in = max(-MAX_MATE_DEPTH, mate_in)
                score_cp = -MAX_SCORE - mate_in * MATE_FACTOR
            else:
                mate_in = min(MAX_MATE_DEPTH, mate_in)
                score_cp = MAX_SCORE - mate_in * MATE_FACTOR
        else:
            score_cp = ev.white().score()
            score_cp = min(score_cp, MAX_NON_MATE_SCORE)
            score_cp = max(score_cp, -MAX_NON_MATE_SCORE)

        # Lichess eval is always from White's perspective - convert to STM
        if not board_turn:  # Black to move
            score_cp = -score_cp

        return score_cp

    def process_game_positions(self, game, max_plys_per_game: int = MAX_PLYS_PER_GAME,
                               opening_plys: int = OPENING_PLYS) -> List[Tuple[chess.Board, int]]:
        """
        Process a single game and return valid positions with centipawn scores.

        This method contains the shared filtering logic used by both training and testing:
        - Skip variant games
        - Skip opening moves
        - Skip game-over positions
        - Skip positions where side-to-move is in check
        - Skip positions after captures (tactically unstable)

        Args:
            game: A chess.pgn game object
            max_plys_per_game: Maximum positions to extract per game
            opening_plys: Number of opening moves to skip

        Returns:
            List of (board_copy, score_cp_stm) tuples where score_cp_stm is the
            centipawn score from side-to-move perspective
        """
        if game is None:
            return []

        # Skip variant games
        if any("Variant" in key for key in game.headers.keys()):
            return []

        positions = []
        board = game.board()
        move_count = 0

        for node in game.mainline():
            move_count += 1
            current_move = node.move

            # Check if this move is a capture BEFORE pushing it
            was_last_move_capture = board.is_capture(current_move)

            if len(positions) >= max_plys_per_game:
                break

            if move_count <= opening_plys:
                board.push(current_move)
                continue

            board.push(current_move)

            if board.is_game_over():
                continue

            # Skip if side to move is in check
            if board.is_check():
                continue

            # Skip if this move was a capture (position after capture is tactically unstable)
            if was_last_move_capture:
                continue

            score_cp = self.eval_to_cp_stm(node.eval(), board.turn == chess.WHITE)
            if score_cp is not None:
                positions.append((board.copy(), score_cp))

        return positions

    def __call__(self, game, max_plys_per_game: int = MAX_PLYS_PER_GAME, opening_plys: int = OPENING_PLYS) -> List[
        Tuple]:
        """
        Process a single game and return positions with evaluations.
        Uses incremental feature updates for efficiency.
        Periodically validates incremental updates against full extraction.
        """
        if game is None:
            return []

        # Skip variant games
        if any("Variant" in key for key in game.headers.keys()):
            return []

        positions = []
        board = game.board()

        # Initialize incremental updater after skipping early moves
        feature_updater = None
        move_count = 0

        for node in game.mainline():
            move_count += 1
            current_move = node.move

            # Check if this move is a capture BEFORE pushing it
            was_last_move_capture = board.is_capture(current_move)

            if len(positions) >= max_plys_per_game:
                break

            if move_count <= opening_plys:
                board.push(current_move)
                continue

            # Initialize feature updater on first position we might use
            if feature_updater is None:
                feature_updater = NNUEIncrementalUpdater(board)

            # Update features incrementally (this also updates the updater's internal board)
            is_white_king_move, is_black_king_move, change_record = feature_updater.update_pre_push(board, node.move)
            board.push(node.move)
            feature_updater.update_post_push(board, is_white_king_move, is_black_king_move, change_record)

            if board.is_game_over():
                continue

            # Skip if side to move is in check
            if board.is_check():
                continue

            # Skip if this move was a capture (position after capture is tactically unstable)
            if was_last_move_capture:
                continue

            score_cp = self.eval_to_cp_stm(node.eval(), board.turn == chess.WHITE)
            if score_cp is not None:
                score_tanh = np.tanh(score_cp / TANH_SCALE)

                if self.nn_type == "NNUE":
                    # Get features from incremental updater (faster than full recompute)
                    white_feat, black_feat = feature_updater.get_features_unsorted()

                    # Periodic validation: compare incremental vs full extraction
                    self.position_count += 1
                    if self.position_count % self.VALIDATION_INTERVAL == 0:
                        if not self.is_matching_full_vs_incremental(board, white_feat, black_feat):
                            # Use full-extraction as fallback
                            white_feat, black_feat = NNUEFeatures.board_to_features(board)

                    stm = 1.0 if board.turn == chess.WHITE else 0.0
                    positions.append((white_feat, black_feat, stm, score_tanh))

                else:  # DNN
                    # Extract features from perspective of side to move
                    feat = DNNFeatures.board_to_features(board)
                    positions.append((feat, score_tanh))

        return positions


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f'{NN_TYPE} Training with Parallel Data Loading')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH',
                        help='Path to checkpoint .pt file to resume training from')
    parser.add_argument('--skip-games', type=int, default=0, metavar='N',
                        help='When resuming, skip the first N games from each pgn.zst file (default: 0)')
    parser.add_argument('--pgn-dir', type=str, default='./pgn',
                        help='Directory containing .pgn.zst files (default: ./pgn)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Total number of epochs to train (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--checkpoint', type=str, default=MODEL_PATH,
                        help=f'Path to save checkpoints (default: {MODEL_PATH})')
    args = parser.parse_args()

    pgn_dir = args.pgn_dir
    resume_checkpoint = None

    print("=" * 60)
    print(f"{NN_TYPE} Training with Parallel Data Loading")
    print("=" * 60)

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Network type: {NN_TYPE}")

    # Create model based on nn_type
    if NN_TYPE == "NNUE":
        model = NNUENetwork(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
    else:  # DNN
        model = DNNNetwork(DNN_INPUT_SIZE)

    # Load checkpoint if resuming
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Checkpoint file not found: {args.resume}")
            exit(1)

        print(f"\nLoading checkpoint from: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # Verify network type matches
        checkpoint_nn_type = resume_checkpoint.get('nn_type', NN_TYPE)
        if checkpoint_nn_type != NN_TYPE:
            print(f"Error: Checkpoint network type ({checkpoint_nn_type}) does not match "
                  f"current configuration ({NN_TYPE})")
            exit(1)

        # Load model weights
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {resume_checkpoint['epoch']} "
              f"with validation loss: {resume_checkpoint['val_loss']:.6f}")

        if 'best_loss' in resume_checkpoint:
            print(f"Best validation loss so far: {resume_checkpoint['best_loss']:.6f}")

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Determine skip_games_count (only used when resuming)
    skip_games_count = args.skip_games if args.resume else 0
    if skip_games_count > 0 and not args.resume:
        print("Warning: --skip-games has no effect without --resume")
        skip_games_count = 0

    # Create trainer
    trainer = ParallelTrainer(
        pgn_dir=pgn_dir,
        model=model,
        batch_size=args.batch_size,
        validation_split=VALIDATION_SPLIT,
        queue_size=QUEUE_MAX_SIZE,
        device=device,
        seed=42,
        skip_games_count=skip_games_count
    )

    # Train with improved parameters
    # - Higher positions_per_epoch for better convergence
    # - Learning rate scheduler to reduce LR when plateauing
    # - Longer patience since LR will be reduced
    # - Gradient clipping for stability
    history = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        positions_per_epoch=POSITIONS_PER_EPOCH,  # 1M positions per epoch (was 100k)
        early_stopping_patience=EARLY_STOPPING_PATIENCE,  # More patience since LR scheduler helps
        checkpoint_path=args.checkpoint,
        lr_scheduler="plateau",  # Reduce LR on plateau
        grad_clip=1.0,  # Gradient clipping
        resume_checkpoint=resume_checkpoint
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Epochs trained: {len(history['train_loss'])}")