import numpy as np
import chess
import chess.pgn
import zstandard as zstd
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from typing import Tuple, List, Iterator, Optional, Dict, Any, Set
import struct
import random
import re
import os
import gc
import time
import glob
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from dataclasses import dataclass, field
from collections import deque
import threading
import ctypes

# Configuration
KING_SQUARES = 64
PIECE_SQUARES = 64
PIECE_TYPES = 5  # P, N, B, R, Q (no King)
COLORS = 2  # White, Black
INPUT_SIZE = KING_SQUARES * PIECE_SQUARES * PIECE_TYPES * COLORS
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1

# Worker configuration
QUEUE_MAX_SIZE = 100  # Max batches in queue
POSITIONS_PER_EPOCH = 1_000_000
BATCH_SIZE = 8192
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 10
QUEUE_READ_TIMEOUT = int(BATCH_SIZE / 512)
LEARNING_RATE = 0.001 #* int(BATCH_SIZE / 512)
VALIDATION_SPLIT = 0.05
GC_INTERVAL = 1000  # Run garbage collection every N batches

TANH_SCALE = 400  # 1200(CP) = 3 = ~pi = 0.99
MAX_SCORE = 10_000  # tanh(10) is almost 1
MATE_FACTOR = 100
MAX_MATE_DEPTH = 10
MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR

# @dataclass
# class WorkerStats:
#     """Statistics for a single worker"""
#     worker_id: int
#     file_path: str
#     games_processed: int = 0
#     positions_extracted: int = 0
#     batches_sent: int = 0
#     file_loops: int = 0
#     wait_time_seconds: float = 0.0
#     processing_time_seconds: float = 0.0
#     last_update: float = field(default_factory=time.time)


# @dataclass
# class MainProcessStats:
#     """Statistics for main process"""
#     batches_consumed: int = 0
#     train_batches: int = 0
#     val_batches: int = 0
#     wait_time_seconds: float = 0.0
#     processing_time_seconds: float = 0.0
#     last_update: float = field(default_factory=time.time)


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


class NNUEFeatures:
    """Handles feature extraction for NNUE network"""

    @staticmethod
    def get_piece_index(piece_type: int, piece_color: bool) -> int:
        """Convert piece type and color to index (0-9)"""
        if piece_type == chess.KING:
            return -1
        type_idx = piece_type - 1
        color_idx = 1 if piece_color else 0
        return type_idx + color_idx * PIECE_TYPES

    @staticmethod
    def get_feature_index(king_sq: int, piece_sq: int, piece_type: int, piece_color: bool) -> int:
        """Calculate the feature index for (king_square, piece_square, piece_type, piece_color)"""
        piece_idx = NNUEFeatures.get_piece_index(piece_type, piece_color)
        if piece_idx == -1:
            return -1
        return king_sq * (PIECE_SQUARES * PIECE_TYPES * COLORS) + \
            piece_sq * (PIECE_TYPES * COLORS) + piece_idx

    @staticmethod
    def flip_square(square: int) -> int:
        """Flip square vertically (A1 <-> A8)"""
        rank = square // 8
        file = square % 8
        return (7 - rank) * 8 + file

    @staticmethod
    def extract_features(board: chess.Board, perspective: bool) -> List[int]:
        """
        Extract active features for one perspective
        perspective: True for white, False for black
        Returns list of active feature indices
        """
        features = []
        king_square = board.king(perspective)
        if king_square is None:
            return features

        if not perspective:  # Black perspective - flip
            king_square = NNUEFeatures.flip_square(king_square)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue

            piece_square = square
            piece_color = piece.color
            piece_type = piece.piece_type

            if not perspective:
                piece_square = NNUEFeatures.flip_square(piece_square)
                piece_color = not piece_color

            feature_idx = NNUEFeatures.get_feature_index(
                king_square, piece_square, piece_type, piece_color
            )

            if feature_idx >= 0:
                features.append(feature_idx)

        return features

    @staticmethod
    def board_to_features(board: chess.Board) -> Tuple[List[int], List[int]]:
        """Extract features for both perspectives"""
        white_features = NNUEFeatures.extract_features(board, chess.WHITE)
        black_features = NNUEFeatures.extract_features(board, chess.BLACK)
        return white_features, black_features


class IncrementalFeatureUpdater:
    """
    Efficiently maintains NNUE features with incremental updates.

    Instead of recomputing all features after each move, this class
    tracks which features need to be added/removed based on the move made.

    For king moves, a full recomputation is needed for that perspective
    since all features depend on the king square.
    """

    def __init__(self, board: chess.Board):
        """Initialize with a board position"""
        self.board = board.copy()
        # Store features as sets for O(1) add/remove
        self.white_features: Set[int] = set(NNUEFeatures.extract_features(board, chess.WHITE))
        self.black_features: Set[int] = set(NNUEFeatures.extract_features(board, chess.BLACK))
        # Cache king squares
        self.white_king_sq = board.king(chess.WHITE)
        self.black_king_sq = board.king(chess.BLACK)

    def _get_feature_for_perspective(self, perspective: bool, piece_sq: int,
                                     piece_type: int, piece_color: bool) -> int:
        """Get feature index for a piece from a given perspective"""
        if perspective:
            king_sq = self.white_king_sq
        else:
            king_sq = NNUEFeatures.flip_square(self.black_king_sq)
            piece_sq = NNUEFeatures.flip_square(piece_sq)
            piece_color = not piece_color

        return NNUEFeatures.get_feature_index(king_sq, piece_sq, piece_type, piece_color)

    def _remove_piece_features(self, square: int, piece_type: int, piece_color: bool):
        """Remove features for a piece at the given square"""
        if piece_type == chess.KING:
            return

        # Remove from white's perspective
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat >= 0:
            self.white_features.discard(white_feat)

        # Remove from black's perspective
        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat >= 0:
            self.black_features.discard(black_feat)

    def _add_piece_features(self, square: int, piece_type: int, piece_color: bool):
        """Add features for a piece at the given square"""
        if piece_type == chess.KING:
            return

        # Add to white's perspective
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat >= 0:
            self.white_features.add(white_feat)

        # Add to black's perspective
        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat >= 0:
            self.black_features.add(black_feat)

    def _recompute_perspective(self, perspective: bool):
        """Fully recompute features for one perspective (needed after king moves)"""
        features = set(NNUEFeatures.extract_features(self.board, perspective))
        if perspective:
            self.white_features = features
        else:
            self.black_features = features

    def push(self, move: chess.Move):
        """
        Update features after making a move.
        Must be called BEFORE board.push(move).
        """
        from_sq = move.from_square
        to_sq = move.to_square

        piece = self.board.piece_at(from_sq)
        if piece is None:
            # Should not happen in valid games
            self.board.push(move)
            return

        moving_piece_type = piece.piece_type
        moving_piece_color = piece.color

        # Check if this is a king move
        is_white_king_move = (moving_piece_type == chess.KING and moving_piece_color == chess.WHITE)
        is_black_king_move = (moving_piece_type == chess.KING and moving_piece_color == chess.BLACK)

        # Check for capture
        captured_piece = self.board.piece_at(to_sq)
        is_en_passant = self.board.is_en_passant(move)

        # Handle en passant capture
        if is_en_passant:
            # The captured pawn is not on to_sq, it's on the en passant square
            ep_sq = to_sq + (-8 if moving_piece_color == chess.WHITE else 8)
            captured_piece = self.board.piece_at(ep_sq)
            if captured_piece:
                self._remove_piece_features(ep_sq, captured_piece.piece_type, captured_piece.color)

        # Remove captured piece features
        if captured_piece and not is_en_passant:
            self._remove_piece_features(to_sq, captured_piece.piece_type, captured_piece.color)

        # Handle castling - need to move the rook too
        is_castling = self.board.is_castling(move)
        if is_castling:
            # Determine rook squares
            if to_sq > from_sq:  # Kingside
                rook_from = chess.H1 if moving_piece_color == chess.WHITE else chess.H8
                rook_to = chess.F1 if moving_piece_color == chess.WHITE else chess.F8
            else:  # Queenside
                rook_from = chess.A1 if moving_piece_color == chess.WHITE else chess.A8
                rook_to = chess.D1 if moving_piece_color == chess.WHITE else chess.D8

            # Remove rook from old square, add to new square
            self._remove_piece_features(rook_from, chess.ROOK, moving_piece_color)
            # Note: rook addition happens after board.push since we need updated king position

        # Remove moving piece from old square (if not king - king features handled by recompute)
        if not is_white_king_move and not is_black_king_move:
            self._remove_piece_features(from_sq, moving_piece_type, moving_piece_color)

        # Make the move on internal board
        self.board.push(move)

        # Update king square cache if king moved
        if is_white_king_move:
            self.white_king_sq = to_sq
        elif is_black_king_move:
            self.black_king_sq = to_sq

        # Handle promotion
        if move.promotion:
            moving_piece_type = move.promotion

        # Add moving piece to new square (if not king)
        if not is_white_king_move and not is_black_king_move:
            self._add_piece_features(to_sq, moving_piece_type, moving_piece_color)

        # Add rook to new square for castling
        if is_castling:
            self._add_piece_features(rook_to, chess.ROOK, moving_piece_color)

        # If king moved, recompute that perspective entirely
        # (all features depend on king square)
        if is_white_king_move:
            self._recompute_perspective(chess.WHITE)
        elif is_black_king_move:
            self._recompute_perspective(chess.BLACK)

    def get_features(self) -> Tuple[List[int], List[int]]:
        """Get current features as sorted lists"""
        return sorted(self.white_features), sorted(self.black_features)

    def get_features_unsorted(self) -> Tuple[List[int], List[int]]:
        """Get current features as lists (faster, no sorting)"""
        return list(self.white_features), list(self.black_features)


class NNUENetwork(nn.Module):
    """PyTorch NNUE Network"""

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE):
        super(NNUENetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ft = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_features, black_features, stm):
        """
        white_features, black_features: dense tensors
        stm: side to move (1.0 for white, 0.0 for black)
        """
        w_hidden = torch.clamp(self.ft(white_features), 0, 1)
        b_hidden = torch.clamp(self.ft(black_features), 0, 1)

        batch_size = white_features.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_size * 2, device=white_features.device)

        # Handle edge case: all same side to move
        white_to_move = (stm > 0.5).squeeze(-1)

        # Ensure white_to_move is 1D boolean tensor
        if white_to_move.dim() == 0:
            white_to_move = white_to_move.unsqueeze(0)

        # Use torch.where for safer indexing that handles edge cases
        n_white = white_to_move.sum().item()
        n_black = batch_size - n_white

        if n_white > 0 and n_black > 0:
            # Mixed batch - normal case
            hidden[white_to_move, :self.hidden_size] = w_hidden[white_to_move]
            hidden[white_to_move, self.hidden_size:] = b_hidden[white_to_move]
            hidden[~white_to_move, :self.hidden_size] = b_hidden[~white_to_move]
            hidden[~white_to_move, self.hidden_size:] = w_hidden[~white_to_move]
        elif n_white > 0:
            # All white to move
            hidden[:, :self.hidden_size] = w_hidden
            hidden[:, self.hidden_size:] = b_hidden
        else:
            # All black to move
            hidden[:, :self.hidden_size] = b_hidden
            hidden[:, self.hidden_size:] = w_hidden

        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return torch.tanh(x)


# def parse_evaluation(comment: str) -> Optional[float]:
#     """
#     Parse evaluation from PGN comment
#     Supports formats:
#     - [%eval 0.24] (centipawn score)
#     - [%eval #3] (mate in 3)
#     - [%eval -1.5] (negative score)
#     """
#     if not comment:
#         return None
#
#     eval_pattern = r'\[%eval\s+(#)?(-?\d+\.?\d*)\]'
#     match = re.search(eval_pattern, comment)
#
#     if match:
#         is_mate = match.group(1) == '#'
#         value = float(match.group(2))
#
#         if is_mate:
#             cp = 10000 if value > 0 else -10000
#         else:
#             cp = value * 100
#
#         return np.tanh(cp / 400.0)
#
#     return None


# def process_game(game, max_positions: int = 120, skip_early: int = 10) -> List[Tuple]:
#     """
#     Process a single game and return positions with evaluations.
#     Uses incremental feature updates for efficiency.
#     """
#     if game is None:
#         return []
#
#     positions = []
#     board = game.board()
#
#     # Initialize incremental updater after skipping early moves
#     feature_updater = None
#     move_count = 0
#
#     for node in game.mainline():
#         move_count += 1
#         current_move = node.move
#
#         # Check if this move is a capture BEFORE pushing it
#         is_capture = board.is_capture(current_move)
#
#         if len(positions) >= max_positions:
#             break
#
#         if move_count <= skip_early:
#             board.push(node.move)
#             continue
#
#         # Initialize feature updater on first position we might use
#         if feature_updater is None:
#             feature_updater = IncrementalFeatureUpdater(board)
#
#         # Update features incrementally (this also updates the updater's internal board)
#         feature_updater.push(node.move)
#         # Keep main board in sync
#         board.push(node.move)
#
#         if board.is_game_over():
#             continue
#
#         if board.is_check():
#             continue
#
#             # Skip if this move was a capture (position after capture is tactically unstable)
#         if is_capture:
#             continue
#
#         #comment = node.comment if hasattr(node, 'comment') else ''
#         #eval_score = parse_evaluation(comment)
#
#         ev = node.eval()
#         if ev is None:
#             score = None
#         else:
#             if ev.is_mate():
#                 mate_in = ev.white().mate()
#                 if mate_in < 0:
#                     score = -10_000
#                 else:
#                     score = 10_000
#             else:
#                 score = ev.white().score()
#                 score = min(score, 10_000)
#                 score = max(score, -10_000)
#
#             score = np.tanh(score / 400)
#
#         if score is not None:
#             # Get features from incremental updater (faster than full recompute)
#             white_feat, black_feat = feature_updater.get_features_unsorted()
#             stm = 1.0 if board.turn == chess.WHITE else 0.0
#
#             # CRITICAL FIX: Lichess eval is always from White's perspective.
#             # But our network outputs from the side-to-move's perspective.
#             # So when Black is to move, we need to negate the evaluation.
#             if board.turn == chess.BLACK:
#                 score = -score
#
#             positions.append((white_feat, black_feat, stm, score))
#
#     return positions


class ProcessGameWithValidation:
    """
    Callable class that processes games with periodic validation of incremental features.
    Each instance maintains its own position counter (no global state).
    """

    VALIDATION_INTERVAL = 10000

    def __init__(self):
        self.position_count = 0

    def __call__(self, game, max_positions: int = 200, skip_early: int = 10) -> List[Tuple]:
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

            if len(positions) >= max_positions:
                break

            if move_count <= skip_early:
                board.push(current_move)
                continue

            # Initialize feature updater on first position we might use
            if feature_updater is None:
                feature_updater = IncrementalFeatureUpdater(board)

            # Update features incrementally (this also updates the updater's internal board)
            feature_updater.push(node.move)
            # Keep main board in sync
            board.push(node.move)

            if board.is_game_over():
                continue

            # Skip if side to move is in check
            if board.is_check():
                continue

            # Skip if this move was a capture (position after capture is tactically unstable)
            if was_last_move_capture:
                continue

            #comment = node.comment if hasattr(node, 'comment') else ''
            #eval_score = parse_evaluation(comment)
            ev = node.eval()
            if ev is None:
                score = None
            else:
                # if ev.is_mate():
                #     mate_in = ev.white().mate()
                #     if mate_in < 0:
                #         score = -10_000
                #     else:
                #         score = 10_000
                # else:
                #     score = ev.white().score()
                #     score = min(score, 10_000)
                #     score = max(score, -10_000)

                if ev.is_mate():
                    mate_in = ev.white().mate()
                    if mate_in < 0:
                        mate_in = max(-MAX_MATE_DEPTH, mate_in)
                        score = -MAX_SCORE - mate_in * MATE_FACTOR
                    else:
                        mate_in = min(MAX_MATE_DEPTH, mate_in)
                        score = MAX_SCORE - mate_in * MATE_FACTOR
                else:
                    score = ev.white().score()
                    score = min(score, MAX_NON_MATE_SCORE)
                    score = max(score, -MAX_NON_MATE_SCORE)

                score = np.tanh(score / TANH_SCALE)

            if score is not None:
                # Get features from incremental updater (faster than full recompute)
                white_feat, black_feat = feature_updater.get_features_unsorted()

                # Periodic validation: compare incremental vs full extraction
                self.position_count += 1
                if self.position_count % self.VALIDATION_INTERVAL == 0:
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

                        # Use full extraction as fallback
                        white_feat, black_feat = white_feat_full, black_feat_full

                stm = 1.0 if board.turn == chess.WHITE else 0.0

                # CRITICAL FIX: Lichess eval is always from White's perspective.
                # But our network outputs from the side-to-move's perspective.
                # So when Black is to move, we need to negate the evaluation.
                if board.turn == chess.BLACK:
                    score = -score

                positions.append((white_feat, black_feat, stm, score))

        return positions


def encode_sparse_batch(positions: List[Tuple]) -> Dict[str, Any]:
    """
    Encode a batch of positions into sparse format for efficient IPC.
    Instead of sending dense tensors, we send indices of non-zero elements.

    Returns dict with:
    - white_indices: list of lists of active indices per sample
    - black_indices: list of lists of active indices per sample
    - stm: numpy array of side-to-move values
    - scores: numpy array of evaluation scores
    - batch_size: number of samples
    """
    white_indices = []
    black_indices = []
    stms = []
    scores = []

    for white_feat, black_feat, stm, score in positions:
        white_indices.append(white_feat)  # Already a list of indices
        black_indices.append(black_feat)
        stms.append(stm)
        scores.append(score)

    return {
        'white_indices': white_indices,
        'black_indices': black_indices,
        'stm': np.array(stms, dtype=np.float32),
        'scores': np.array(scores, dtype=np.float32),
        'batch_size': len(positions)
    }


def decode_sparse_batch(batch_data: Dict[str, Any], device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
    """
    Decode sparse batch data back to dense tensors.
    Called in main process after receiving from queue.
    """
    batch_size = batch_data['batch_size']
    white_indices = batch_data['white_indices']
    black_indices = batch_data['black_indices']

    # Create dense tensors
    white_input = torch.zeros(batch_size, INPUT_SIZE, device=device)
    black_input = torch.zeros(batch_size, INPUT_SIZE, device=device)

    for i in range(batch_size):
        if white_indices[i]:
            white_input[i, white_indices[i]] = 1.0
        if black_indices[i]:
            black_input[i, black_indices[i]] = 1.0

    stm = torch.tensor(batch_data['stm'], dtype=torch.float32, device=device).unsqueeze(1)
    scores = torch.tensor(batch_data['scores'], dtype=torch.float32, device=device).unsqueeze(1)

    return white_input, black_input, stm, scores


def worker_process(
        worker_id: int,
        pgn_file: str,
        output_queue: Queue,
        stop_event: Event,
        stats: SharedStats,
        batch_size: int = BATCH_SIZE,
        max_positions_per_game: int = 200,
        skip_early_moves: int = 10,
        shuffle_buffer_size: int = SHUFFLE_BUFFER_SIZE #2000  # Reduced from 10000 for faster startup
):
    """
    Worker process that streams positions from a PGN file.
    Loops through the file repeatedly until stop_event is set.
    Uses a shuffle buffer to randomize position order.
    """
    print(f"Worker {worker_id} starting: {os.path.basename(pgn_file)}")

    position_buffer = []
    gc_counter = 0

    # Create local game processor with validation (no shared state)
    game_processor = ProcessGameWithValidation()

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
                            with stats.worker_file_loops[worker_id].get_lock():
                                stats.worker_file_loops[worker_id].value += 1
                            break

                        # Process game with periodic validation (using local counter)
                        positions = game_processor(game, max_positions_per_game, skip_early_moves)

                        if positions:
                            position_buffer.extend(positions)
                            with stats.worker_games[worker_id].get_lock():
                                stats.worker_games[worker_id].value += 1
                            with stats.worker_positions[worker_id].get_lock():
                                stats.worker_positions[worker_id].value += len(positions)

                        process_time = time.time() - process_start
                        with stats.worker_process_ms[worker_id].get_lock():
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
                                        with stats.worker_batches[worker_id].get_lock():
                                            stats.worker_batches[worker_id].value += 1
                                        break
                                    except:
                                        # Queue full
                                        with stats.queue_full_count.get_lock():
                                            stats.queue_full_count.value += 1

                                wait_time = time.time() - wait_start
                                with stats.worker_wait_ms[worker_id].get_lock():
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

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True,
                 checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

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
        }
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
            seed: int = 42
    ):
        self.pgn_dir = pgn_dir
        self.model = model.to(device)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.queue_size = queue_size
        self.device = device
        self.seed = seed

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
                      self.batch_size)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

        print(f"Started {len(self.workers)} worker processes")

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
                with self.stats.main_wait_ms.get_lock():
                    self.stats.main_wait_ms.value += int(wait_time * 1000)
                return batch
            except:
                with self.stats.queue_empty_count.get_lock():
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
            #is_validation = self.rng.random() < self.validation_split
            is_validation = self.stats.main_val_batches.value < positions_per_epoch / BATCH_SIZE * self.validation_split

            with self.stats.main_batches.get_lock():
                self.stats.main_batches.value += 1

            if is_validation:
                # Store for validation (with size limit via deque maxlen)
                with self.val_buffer_lock:
                    self.val_buffer.append(batch_data)
                with self.stats.main_val_batches.get_lock():
                    self.stats.main_val_batches.value += 1
            else:
                # Training step
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

                with self.stats.main_train_batches.get_lock():
                    self.stats.main_train_batches.value += 1

                # Clean up tensors
                del white_input, black_input, stm, target, output, loss

            # Clean up batch data
            del batch_data

            process_time = time.time() - process_start
            with self.stats.main_process_ms.get_lock():
                self.stats.main_process_ms.value += int(process_time * 1000)

            # Progress update
            if train_batch_count != 0 and train_batch_count % int(POSITIONS_PER_EPOCH / BATCH_SIZE / 5) == 0: # 5 prints per epoch
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
        with self.stats.main_val_batches.get_lock():
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
                white_input, black_input, stm, target = decode_sparse_batch(
                    batch_data, self.device
                )

                output = self.model(white_input, black_input, stm)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                batch_count += 1

                # Clean up
                del white_input, black_input, stm, target, output, loss

        # Clear the local copy
        del val_batches

        self.model.train()
        return total_loss / max(1, batch_count)

    def train(
            self,
            epochs: int = 50,
            lr: float = LEARNING_RATE,
            positions_per_epoch: int = 100000,
            early_stopping_patience: int = 5,
            checkpoint_path: str = "best_model.pt",
            lr_scheduler: str = "plateau",  # "plateau", "step", or "none"
            grad_clip: float = 1.0  # Gradient clipping max norm
    ) -> Dict[str, List[float]]:
        """Main training loop with LR scheduling and gradient clipping"""

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Setup learning rate scheduler
        scheduler = None
        if lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5,
                min_lr=1e-6
            )
        elif lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.5
            )

        self.grad_clip = grad_clip

        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            checkpoint_path=checkpoint_path
        )

        history = {'train_loss': [], 'val_loss': [], 'lr': []}

        self.start_workers()

        try:
            for epoch in range(epochs):
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


class NNUEInference:
    """Numpy-based inference engine"""

    def __init__(self, model: NNUENetwork):
        model.eval()
        self.ft_weight = model.ft.weight.detach().cpu().numpy()
        self.ft_bias = model.ft.bias.detach().cpu().numpy()
        self.l1_weight = model.l1.weight.detach().cpu().numpy()
        self.l1_bias = model.l1.bias.detach().cpu().numpy()
        self.l2_weight = model.l2.weight.detach().cpu().numpy()
        self.l2_bias = model.l2.bias.detach().cpu().numpy()
        self.l3_weight = model.l3.weight.detach().cpu().numpy()
        self.l3_bias = model.l3.bias.detach().cpu().numpy()

    def clipped_relu(self, x):
        return np.clip(x, 0, 1)

    def evaluate(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        white_input = np.zeros(INPUT_SIZE)
        black_input = np.zeros(INPUT_SIZE)

        for f in white_features:
            white_input[f] = 1.0
        for f in black_features:
            black_input[f] = 1.0

        white_hidden = self.clipped_relu(np.dot(white_input, self.ft_weight.T) + self.ft_bias)
        black_hidden = self.clipped_relu(np.dot(black_input, self.ft_weight.T) + self.ft_bias)

        if stm:
            hidden = np.concatenate([white_hidden, black_hidden])
        else:
            hidden = np.concatenate([black_hidden, white_hidden])

        x = self.clipped_relu(np.dot(hidden, self.l1_weight.T) + self.l1_bias)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        output = np.dot(x, self.l3_weight.T) + self.l3_bias

        return np.tanh(output[0])

    def evaluate_board(self, board: chess.Board) -> float:
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate(white_feat, black_feat, board.turn == chess.WHITE)

    def save_weights(self, filename: str):
        with open(filename, 'wb') as f:
            f.write(struct.pack('III', INPUT_SIZE, HIDDEN_SIZE, 32))
            self.ft_weight.tofile(f)
            self.ft_bias.tofile(f)
            self.l1_weight.tofile(f)
            self.l1_bias.tofile(f)
            self.l2_weight.tofile(f)
            self.l2_bias.tofile(f)
            self.l3_weight.tofile(f)
            self.l3_bias.tofile(f)

    @classmethod
    def load_weights(cls, filename: str):
        with open(filename, 'rb') as f:
            input_size, hidden_size, l1_size = struct.unpack('III', f.read(12))
            model = NNUENetwork(input_size, hidden_size)
            inference = cls(model)

            inference.ft_weight = np.fromfile(f, dtype=np.float32,
                                              count=hidden_size * input_size).reshape(hidden_size, input_size)
            inference.ft_bias = np.fromfile(f, dtype=np.float32, count=hidden_size)
            inference.l1_weight = np.fromfile(f, dtype=np.float32,
                                              count=l1_size * hidden_size * 2).reshape(l1_size, hidden_size * 2)
            inference.l1_bias = np.fromfile(f, dtype=np.float32, count=l1_size)
            inference.l2_weight = np.fromfile(f, dtype=np.float32,
                                              count=l1_size * l1_size).reshape(l1_size, l1_size)
            inference.l2_bias = np.fromfile(f, dtype=np.float32, count=l1_size)
            inference.l3_weight = np.fromfile(f, dtype=np.float32,
                                              count=l1_size).reshape(1, l1_size)
            inference.l3_bias = np.fromfile(f, dtype=np.float32, count=1)

            return inference


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)

    pgn_dir = "./pgn"

    print("=" * 60)
    print("NNUE Training with Parallel Data Loading")
    print("=" * 60)

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = NNUENetwork()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = ParallelTrainer(
        pgn_dir=pgn_dir,
        model=model,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        queue_size=QUEUE_MAX_SIZE,
        device=device,
        seed=42
    )

    # Train with improved parameters
    # - Higher positions_per_epoch for better convergence
    # - Learning rate scheduler to reduce LR when plateauing
    # - Longer patience since LR will be reduced
    # - Gradient clipping for stability
    history = trainer.train(
        epochs=5000,
        lr=LEARNING_RATE,
        positions_per_epoch=POSITIONS_PER_EPOCH,  # 1M positions per epoch (was 100k)
        early_stopping_patience=100,  # More patience since LR scheduler helps
        checkpoint_path="best_model.pt",
        lr_scheduler="plateau",  # Reduce LR on plateau
        grad_clip=1.0  # Gradient clipping
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Epochs trained: {len(history['train_loss'])}")

    # Create inference engine and test
    print("\nCreating inference engine...")
    inference = NNUEInference(model)

    test_board = chess.Board()
    eval_score = inference.evaluate_board(test_board)
    print(f"Starting position evaluation: {eval_score:.4f}")
    print(f"Centipawn equivalent: {np.arctanh(np.clip(eval_score, -0.99, 0.99)) * TANH_SCALE:.1f}")

    # Save weights
    inference.save_weights("nnue_weights.bin")
    print("Weights saved to nnue_weights.bin")

    # Save history
    import json

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training history saved to training_history.json")