"""
Neural Network Inference Module for Chess Engine
PHASE 2 OPTIMIZED VERSION

Optimizations:
- Uses Cython-accelerated ops when available (5-10x faster)
- Pre-allocated buffers for all intermediate computations
- Inlined clipped_relu operations
- Batched accumulator updates
- Pre-computed lookup tables
"""

import sys
from typing import List, Tuple, Set, Dict, Optional
import chess
from cached_board import CachedBoard
import numpy as np
import torch
from torch import nn as nn

# Try to import Cython-optimized operations
try:
    from nn_ops_fast import (
        dnn_evaluate_incremental as _cy_dnn_eval,
        nnue_evaluate_incremental as _cy_nnue_eval,
        nnue_evaluate_incremental_int8 as _cy_nnue_eval_int8,
        nnue_evaluate_incremental_int16 as _cy_nnue_eval_int16,
        dnn_update_accumulator as _cy_dnn_update,
        nnue_update_accumulator as _cy_nnue_update,
        clipped_relu_inplace,
        clipped_relu_copy,
        get_piece_index as _cy_get_piece_index,
        get_nnue_feature_index as _cy_get_nnue_feature_index,
        flip_square as _cy_flip_square,
        get_dnn_feature_index as _cy_get_dnn_feature_index,
    )
    HAS_CYTHON = True
    print("✓ Using Cython-accelerated NN operations")
except ImportError as e:
    from nn_ops_fallback import (
        dnn_evaluate_incremental as _cy_dnn_eval,
        nnue_evaluate_incremental as _cy_nnue_eval,
        nnue_evaluate_incremental_int8 as _cy_nnue_eval_int8,
        nnue_evaluate_incremental_int16 as _cy_nnue_eval_int16,
        dnn_update_accumulator as _cy_dnn_update,
        nnue_update_accumulator as _cy_nnue_update,
        clipped_relu_inplace,
        clipped_relu_copy,
        get_piece_index as _cy_get_piece_index,
        get_nnue_feature_index as _cy_get_nnue_feature_index,
        flip_square as _cy_flip_square,
        get_dnn_feature_index as _cy_get_dnn_feature_index,
    )
    HAS_CYTHON = False
    #print("! Cython not available, using pure Python fallback")
    print(e)
    print("\033[91m! Cython not available, using pure Python fallback\033[0m")

L1_QUANTIZATION = "NONE" # Options: "NONE" (FP32), "INT8", "INT16"

KING_SQUARES = 64
PIECE_SQUARES = 64
PIECE_TYPES = 5  # P, N, B, R, Q (no King)
COLORS = 2  # White, Black
NNUE_INPUT_SIZE = KING_SQUARES * PIECE_SQUARES * PIECE_TYPES * COLORS
NNUE_HIDDEN_SIZE = 256
DNN_INPUT_SIZE = 768  # 64 squares * 6 piece types * 2 colors
DNN_HIDDEN_LAYERS = [1024, 256, 32]
OUTPUT_SIZE = 1
TANH_SCALE = 410 # Stockfish value
MAX_SCORE = 10_000

# Pre-computed lookup tables
_FLIPPED_SQUARES = np.array([(7 - (sq // 8)) * 8 + (sq % 8) for sq in range(64)], dtype=np.int32)

# Piece type mapping (shared by DNN and NNUE):
# Uses piece_type - 1 since chess.PAWN=1, chess.KNIGHT=2, chess.BISHOP=3, chess.ROOK=4, chess.QUEEN=5, chess.KING=6
# This gives: P=0, N=1, B=2, R=3, Q=4, K=5
#
# DNN encoding (768 features):
#   - 12 planes × 64 squares
#   - Planes 0-5: STM pieces [P, N, B, R, Q, K], Planes 6-11: Opponent pieces
#   - Feature index = piece_idx * 64 + square
#
# NNUE encoding (40,960 features):
#   - King excluded from piece features (only P, N, B, R, Q)
#   - Feature index = king_sq * 640 + piece_sq * 10 + (type_idx + color_idx * 5)


class NNUEFeatures:
    """Handles feature extraction for NNUE network"""

    @staticmethod
    def get_piece_index(piece_type: int, is_friendly_piece: bool) -> int:
        if piece_type == chess.KING:
            return -1
        type_idx = piece_type - 1
        # Enemy features first
        color_idx = 1 if is_friendly_piece else 0
        return type_idx + color_idx * PIECE_TYPES

    @staticmethod
    def get_feature_index(king_sq: int, piece_sq: int, piece_type: int, is_friendly_piece: bool) -> int:
        piece_idx = NNUEFeatures.get_piece_index(piece_type, is_friendly_piece)
        if piece_idx == -1:
            return -1
        return king_sq * 640 + piece_sq * 10 + piece_idx

    @staticmethod
    def flip_square(square: int) -> int:
        return int(_FLIPPED_SQUARES[square])

    @staticmethod
    def extract_features(board: CachedBoard, perspective: bool) -> List[int]:
        features = []
        king_square = board.king(perspective)
        if king_square is None:
            return features

        if not perspective:
            king_square = int(_FLIPPED_SQUARES[king_square])

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue

            piece_square = square
            is_friendly_piece = piece.color
            piece_type = piece.piece_type

            if not perspective:
                piece_square = int(_FLIPPED_SQUARES[piece_square])
                is_friendly_piece = not is_friendly_piece

            feature_idx = NNUEFeatures.get_feature_index(
                king_square, piece_square, piece_type, is_friendly_piece
            )

            if feature_idx >= 0:
                features.append(feature_idx)

        return features

    @staticmethod
    def board_to_features(board: CachedBoard) -> Tuple[List[int], List[int]]:
        white_features = NNUEFeatures.extract_features(board, chess.WHITE)
        black_features = NNUEFeatures.extract_features(board, chess.BLACK)
        return white_features, black_features


class DNNFeatures:
    """
    Handles feature extraction for DNN network (768-dimensional one-hot encoding).

    Encoding scheme:
    - 12 planes × 64 squares = 768 features
    - Planes 0-5: STM (side-to-move) pieces [P, N, B, R, Q, K]
    - Planes 6-11: Opponent pieces [P, N, B, R, Q, K]
    - Piece order matches NNUE (piece_type - 1): P=0, N=1, B=2, R=3, Q=4, K=5
    - Board is flipped when Black to move (so STM always sees board from their perspective)
    - Feature index = piece_idx * 64 + square
    """

    @staticmethod
    def get_piece_index(piece_type: int, is_friendly_piece: bool) -> int:
        """Get piece index (0-11) for the feature encoding.

        Uses piece_type - 1 to match NNUE ordering: P=0, N=1, B=2, R=3, Q=4, K=5
        """
        type_idx = piece_type - 1  # chess.PAWN=1 -> 0, chess.KNIGHT=2 -> 1, etc.
        return type_idx + (0 if is_friendly_piece else 6)

    @staticmethod
    def extract_features(board: CachedBoard, perspective: bool) -> List[int]:
        features = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            adj_square = square
            if not perspective:
                adj_square = int(_FLIPPED_SQUARES[square])

            is_friendly_piece = (piece.color == chess.WHITE) == perspective
            piece_idx = (piece.piece_type - 1) + (0 if is_friendly_piece else 6)
            feature_idx = piece_idx * 64 + adj_square
            features.append(feature_idx)

        return features

    @staticmethod
    def board_to_features(board: CachedBoard) -> List[int]:
        return DNNFeatures.extract_features(board, board.turn == chess.WHITE)


class NNUEIncrementalUpdater:
    """Efficiently maintains NNUE features with incremental updates and undo support."""

    def __init__(self, board: CachedBoard):
        self.white_features: Set[int] = set(NNUEFeatures.extract_features(board, chess.WHITE))
        self.black_features: Set[int] = set(NNUEFeatures.extract_features(board, chess.BLACK))
        self.white_king_sq = board.king(chess.WHITE)
        self.black_king_sq = board.king(chess.BLACK)
        self.history_stack: List[Dict] = []
        self.last_change: Optional[Dict] = None

    def _get_feature_for_perspective(self, perspective: bool, piece_sq: int,
                                     piece_type: int, is_friendly_piece: bool) -> int:
        if perspective:
            king_sq = self.white_king_sq
        else:
            king_sq = int(_FLIPPED_SQUARES[self.black_king_sq])
            piece_sq = int(_FLIPPED_SQUARES[piece_sq])
            is_friendly_piece = not is_friendly_piece
        return NNUEFeatures.get_feature_index(king_sq, piece_sq, piece_type, is_friendly_piece)

    def _remove_piece_features(self, square: int, piece_type: int, piece_color: bool,
                               change_record: Dict):
        if piece_type == chess.KING:
            return

        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat >= 0 and white_feat in self.white_features:
            self.white_features.discard(white_feat)
            change_record['white_removed'].add(white_feat)

        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat >= 0 and black_feat in self.black_features:
            self.black_features.discard(black_feat)
            change_record['black_removed'].add(black_feat)

    def _add_piece_features(self, square: int, piece_type: int, piece_color: bool,
                            change_record: Dict):
        if piece_type == chess.KING:
            return

        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat >= 0 and white_feat not in self.white_features:
            self.white_features.add(white_feat)
            change_record['white_added'].add(white_feat)

        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat >= 0 and black_feat not in self.black_features:
            self.black_features.add(black_feat)
            change_record['black_added'].add(black_feat)

    def update_pre_push(self, board_before_push: CachedBoard, move: chess.Move) -> Tuple[bool, bool, Dict]:
        """
        Phase 1 of two-phase push. Call BEFORE board.push(move).

        This method handles all feature changes that can be computed from the pre-move board:
        - Captures (including en passant)
        - Castling rook movements
        - Non-king piece movements

        For king moves, it saves the previous features for undo but defers the feature
        rebuild to update_post_push() since all features depend on the king square.

        Returns:
            Tuple of (is_white_king_move, is_black_king_move, change_record)
            Pass these to update_post_push() after calling board.push(move).
        """
        from_sq = move.from_square
        to_sq = move.to_square

        change_record = {
            'white_added': set(), 'white_removed': set(),
            'black_added': set(), 'black_removed': set(),
            'white_king_moved': False, 'black_king_moved': False,
            'prev_white_king_sq': self.white_king_sq,
            'prev_black_king_sq': self.black_king_sq,
            'prev_white_features': None, 'prev_black_features': None,
        }

        piece = board_before_push.piece_at(from_sq)
        if piece is None:
            return False, False, change_record

        moving_piece_type = piece.piece_type
        moving_piece_color = piece.color

        is_white_king_move = (moving_piece_type == chess.KING and moving_piece_color == chess.WHITE)
        is_black_king_move = (moving_piece_type == chess.KING and moving_piece_color == chess.BLACK)

        # For king moves, save the entire feature set for that perspective (for undo)
        # and update the king square. Feature rebuild happens in update_post_push.
        if is_white_king_move:
            change_record['white_king_moved'] = True
            change_record['prev_white_features'] = self.white_features.copy()
            self.white_king_sq = to_sq
        if is_black_king_move:
            change_record['black_king_moved'] = True
            change_record['prev_black_features'] = self.black_features.copy()
            self.black_king_sq = to_sq

        # Handle captures - these affect the non-moving side's features
        captured_piece = board_before_push.piece_at(to_sq)
        is_en_passant = board_before_push.is_en_passant(move)

        if is_en_passant:
            ep_sq = to_sq + (-8 if moving_piece_color == chess.WHITE else 8)
            captured_piece = board_before_push.piece_at(ep_sq)
            if captured_piece:
                self._remove_piece_features(ep_sq, captured_piece.piece_type,
                                            captured_piece.color, change_record)

        if captured_piece and not is_en_passant:
            self._remove_piece_features(to_sq, captured_piece.piece_type,
                                        captured_piece.color, change_record)

        # Handle castling rook movement
        is_castling = board_before_push.is_castling(move)
        rook_to = None
        if is_castling:
            if to_sq > from_sq:  # Kingside
                rook_from = chess.H1 if moving_piece_color == chess.WHITE else chess.H8
                rook_to = chess.F1 if moving_piece_color == chess.WHITE else chess.F8
            else:  # Queenside
                rook_from = chess.A1 if moving_piece_color == chess.WHITE else chess.A8
                rook_to = chess.D1 if moving_piece_color == chess.WHITE else chess.D8
            self._remove_piece_features(rook_from, chess.ROOK, moving_piece_color, change_record)

        # For non-king moves, handle the moving piece's feature changes
        if not is_white_king_move and not is_black_king_move:
            self._remove_piece_features(from_sq, moving_piece_type, moving_piece_color, change_record)

            final_piece_type = move.promotion if move.promotion else moving_piece_type
            self._add_piece_features(to_sq, final_piece_type, moving_piece_color, change_record)

        # For castling, add the rook at its new position
        if is_castling and rook_to is not None:
            self._add_piece_features(rook_to, chess.ROOK, moving_piece_color, change_record)

        return is_white_king_move, is_black_king_move, change_record

    def update_post_push(self, board_after_push: CachedBoard,
                         is_white_king_move: bool,
                         is_black_king_move: bool,
                         change_record: Dict):
        """
        Phase 2 of two-phase push. Call AFTER board.push(move).

        For king moves, this rebuilds the entire feature set for that perspective
        since all features are indexed by the king square.

        Args:
            board_after_push: Board state after the move was pushed
            is_white_king_move: From update_pre_push return value
            is_black_king_move: From update_pre_push return value
            change_record: From update_pre_push return value
        """
        # For king moves, rebuild the entire feature set for that perspective
        # because all features are indexed by the king square
        if is_white_king_move:
            new_white_features = set(NNUEFeatures.extract_features(board_after_push, chess.WHITE))
            # Update the change record for proper accumulator updates
            prev_features = change_record.get('prev_white_features') or set()
            change_record['white_added'] = new_white_features - prev_features
            change_record['white_removed'] = prev_features - new_white_features
            self.white_features = new_white_features

        if is_black_king_move:
            new_black_features = set(NNUEFeatures.extract_features(board_after_push, chess.BLACK))
            prev_features = change_record.get('prev_black_features') or set()
            change_record['black_added'] = new_black_features - prev_features
            change_record['black_removed'] = prev_features - new_black_features
            self.black_features = new_black_features

        self.history_stack.append(change_record)
        self.last_change = change_record

    def push(self, board_before_push: CachedBoard, move: chess.Move):
        """
        Single-phase push for backward compatibility.
        Combines update_pre_push and update_post_push.

        Note: This creates a temporary board copy. For better performance,
        use the two-phase API (update_pre_push / update_post_push) directly.
        """
        is_white_king_move, is_black_king_move, change_record = self.update_pre_push(
            board_before_push, move
        )

        # Create temporary board for post-push state
        temp_board = board_before_push.copy()
        temp_board.push(move)

        self.update_post_push(temp_board, is_white_king_move, is_black_king_move, change_record)

    def pop(self) -> Dict:
        if not self.history_stack:
            raise ValueError("No moves to pop")

        change_record = self.history_stack.pop()

        if change_record['white_king_moved']:
            self.white_features = change_record['prev_white_features']
            self.white_king_sq = change_record['prev_white_king_sq']
        else:
            self.white_features -= change_record['white_added']
            self.white_features |= change_record['white_removed']

        if change_record['black_king_moved']:
            self.black_features = change_record['prev_black_features']
            self.black_king_sq = change_record['prev_black_king_sq']
        else:
            self.black_features -= change_record['black_added']
            self.black_features |= change_record['black_removed']

        self.last_change = change_record
        return change_record

    def get_features_unsorted(self) -> Tuple[Set[int], Set[int]]:
        return self.white_features, self.black_features

    def get_features(self, board: CachedBoard) -> List[int]:
        if board.turn == chess.WHITE:
            return list(self.white_features)
        return list(self.black_features)

    def get_features_both(self) -> Tuple[List[int], List[int]]:
        return list(self.white_features), list(self.black_features)

    def clear_history(self):
        self.history_stack.clear()

    def history_size(self) -> int:
        return len(self.history_stack)


class DNNIncrementalUpdater:
    """Incrementally maintains DNN features for both perspectives with efficient undo support."""

    def __init__(self, board: CachedBoard):
        self.white_features: Set[int] = set(DNNFeatures.extract_features(board, chess.WHITE))
        self.black_features: Set[int] = set(DNNFeatures.extract_features(board, chess.BLACK))
        self.history_stack: List[Dict[str, Set[int]]] = []
        self.last_change: Optional[Dict[str, Set[int]]] = None

    @staticmethod
    def _get_feature_for_perspective(perspective: bool, square: int,
                                     piece_type: int, piece_color: bool) -> int:
        """Get DNN feature index for a piece from a given perspective.

        New encoding: feature_idx = piece_idx * 64 + oriented_square
        Piece order: P=0, N=1, B=2, R=3, Q=4, K=5 (piece_type - 1)
        """
        if not perspective:
            square = int(_FLIPPED_SQUARES[square])
        is_friendly_piece = (piece_color == chess.WHITE) == perspective
        # piece_type - 1 gives: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
        piece_idx = (piece_type - 1) + (0 if is_friendly_piece else 6)
        return piece_idx * 64 + square

    def _remove_piece_features(self, square: int, piece_type: int, piece_color: bool,
                               change_record: Dict[str, Set[int]]):
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat in self.white_features:
            self.white_features.discard(white_feat)
            change_record['white_removed'].add(white_feat)

        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat in self.black_features:
            self.black_features.discard(black_feat)
            change_record['black_removed'].add(black_feat)

    def _add_piece_features(self, square: int, piece_type: int, piece_color: bool,
                            change_record: Dict[str, Set[int]]):
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat not in self.white_features:
            self.white_features.add(white_feat)
            change_record['white_added'].add(white_feat)

        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat not in self.black_features:
            self.black_features.add(black_feat)
            change_record['black_added'].add(black_feat)

    def push(self, board_before_push: CachedBoard, move: chess.Move):
        from_sq = move.from_square
        to_sq = move.to_square

        change_record = {
            'white_added': set(), 'white_removed': set(),
            'black_added': set(), 'black_removed': set()
        }

        piece = board_before_push.piece_at(from_sq)
        if piece is None:
            self.history_stack.append(change_record)
            self.last_change = change_record
            return

        moving_piece_type = piece.piece_type
        moving_piece_color = piece.color

        captured_piece = board_before_push.piece_at(to_sq)
        is_en_passant = board_before_push.is_en_passant(move)

        if is_en_passant:
            ep_sq = to_sq + (-8 if moving_piece_color == chess.WHITE else 8)
            captured_piece = board_before_push.piece_at(ep_sq)
            if captured_piece:
                self._remove_piece_features(ep_sq, captured_piece.piece_type,
                                            captured_piece.color, change_record)

        if captured_piece and not is_en_passant:
            self._remove_piece_features(to_sq, captured_piece.piece_type,
                                        captured_piece.color, change_record)

        is_castling = board_before_push.is_castling(move)
        rook_to = None
        if is_castling:
            if to_sq > from_sq:
                rook_from = chess.H1 if moving_piece_color == chess.WHITE else chess.H8
                rook_to = chess.F1 if moving_piece_color == chess.WHITE else chess.F8
            else:
                rook_from = chess.A1 if moving_piece_color == chess.WHITE else chess.A8
                rook_to = chess.D1 if moving_piece_color == chess.WHITE else chess.D8
            self._remove_piece_features(rook_from, chess.ROOK, moving_piece_color, change_record)

        self._remove_piece_features(from_sq, moving_piece_type, moving_piece_color, change_record)

        if move.promotion:
            moving_piece_type = move.promotion

        self._add_piece_features(to_sq, moving_piece_type, moving_piece_color, change_record)

        if is_castling:
            self._add_piece_features(rook_to, chess.ROOK, moving_piece_color, change_record)

        self.history_stack.append(change_record)
        self.last_change = change_record

    def pop(self) -> Dict[str, Set[int]]:
        if not self.history_stack:
            raise ValueError("No moves to pop - history stack is empty")

        change_record = self.history_stack.pop()
        self.white_features -= change_record['white_added']
        self.white_features |= change_record['white_removed']
        self.black_features -= change_record['black_added']
        self.black_features |= change_record['black_removed']
        self.last_change = change_record
        return change_record

    def get_last_change(self) -> Optional[Dict[str, Set[int]]]:
        return self.last_change

    def get_features_both(self) -> Tuple[List[int], List[int]]:
        return list(self.white_features), list(self.black_features)


class NNUENetwork(nn.Module):
    """PyTorch NNUE Network"""

    def __init__(self, input_size=NNUE_INPUT_SIZE, hidden_size=NNUE_HIDDEN_SIZE):
        super(NNUENetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ft = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_features, black_features, stm_white):
        white_hidden = torch.clamp(white_features, 0, 1)
        black_hidden = torch.clamp(black_features, 0, 1)
        hidden = torch.where(
            stm_white.unsqueeze(-1),
            torch.cat([white_hidden, black_hidden], dim=-1),
            torch.cat([black_hidden, white_hidden], dim=-1)
        )
        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return x


class DNNNetwork(nn.Module):
    """PyTorch DNN Network for chess position evaluation"""

    def __init__(self, input_size=DNN_INPUT_SIZE, hidden_layers=None):
        super(DNNNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.l1 = nn.Linear(input_size, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, features):
        x = torch.clamp(self.l1(features), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.clamp(self.l3(x), 0, 1)
        x = self.l4(x)
        return x


class NNUEInference:
    """
    PHASE 2 OPTIMIZED: Numpy-based inference engine for NNUE.
    Uses Cython-accelerated operations when available.

    Supports L1 layer quantization (INT8/INT16) controlled by L1_QUANTIZATION global.
    """

    def __init__(self, model: NNUENetwork):
        model.eval()
        with torch.no_grad():
            self.ft_weight = model.ft.weight.cpu().numpy().astype(np.float32)
            self.ft_bias = model.ft.bias.cpu().numpy().astype(np.float32)
            self.l1_weight = model.l1.weight.cpu().numpy().astype(np.float32)
            self.l1_bias = model.l1.bias.cpu().numpy().astype(np.float32)
            self.l2_weight = model.l2.weight.cpu().numpy().astype(np.float32)
            self.l2_bias = model.l2.bias.cpu().numpy().astype(np.float32)
            self.l3_weight = model.l3.weight.cpu().numpy().astype(np.float32)
            self.l3_bias = model.l3.bias.cpu().numpy().astype(np.float32)

        self.hidden_size = model.hidden_size
        self.white_accumulator = None
        self.black_accumulator = None

        # Pre-allocate all buffers
        self._hidden_buf = np.empty(self.hidden_size * 2, dtype=np.float32)
        self._l1_buf = np.empty(self.l1_bias.shape[0], dtype=np.float32)
        self._l2_buf = np.empty(self.l2_bias.shape[0], dtype=np.float32)
        self._white_clipped = np.empty(self.hidden_size, dtype=np.float32)
        self._black_clipped = np.empty(self.hidden_size, dtype=np.float32)
        self._max_feature_idx = self.ft_weight.shape[1]

        # L1 Quantization setup
        self._quantization_mode = L1_QUANTIZATION
        self._l1_weight_q = None
        self._l1_combined_scale = None
        self._hidden_buf_q = None

        if L1_QUANTIZATION == "INT8":
            self._setup_int8_quantization()
        elif L1_QUANTIZATION == "INT16":
            self._setup_int16_quantization()

    def _setup_int8_quantization(self):
        """Initialize INT8 quantization for L1 layer."""
        # Quantize L1 weights: symmetric quantization to [-127, 127]
        weight_abs_max = np.max(np.abs(self.l1_weight))
        weight_scale = weight_abs_max / 127.0 if weight_abs_max > 0 else 1.0
        self._l1_weight_q = np.clip(
            np.round(self.l1_weight / weight_scale), -127, 127
        ).astype(np.int8)

        # Input scale: hidden_buf is in [0, 1], scale to [0, 127]
        input_scale = 1.0 / 127.0

        # Pre-compute combined scale for dequantization
        self._l1_combined_scale = np.float32(input_scale * weight_scale)

        # Pre-allocate quantized input buffer
        self._hidden_buf_q = np.empty(self.hidden_size * 2, dtype=np.int8)

        print(f"  L1 INT8 quantization: weight_scale={weight_scale:.6f}, combined_scale={self._l1_combined_scale:.8f}")

    def _setup_int16_quantization(self):
        """Initialize INT16 quantization for L1 layer."""
        # Quantize L1 weights: symmetric quantization to [-32767, 32767]
        weight_abs_max = np.max(np.abs(self.l1_weight))
        weight_scale = weight_abs_max / 32767.0 if weight_abs_max > 0 else 1.0
        self._l1_weight_q = np.clip(
            np.round(self.l1_weight / weight_scale), -32767, 32767
        ).astype(np.int16)

        # Input scale: hidden_buf is in [0, 1], scale to [0, 32767]
        input_scale = 1.0 / 32767.0

        # Pre-compute combined scale for dequantization
        self._l1_combined_scale = np.float32(input_scale * weight_scale)

        # Pre-allocate quantized input buffer
        self._hidden_buf_q = np.empty(self.hidden_size * 2, dtype=np.int16)

        print(f"  L1 INT16 quantization: weight_scale={weight_scale:.6f}, combined_scale={self._l1_combined_scale:.10f}")

    def evaluate_full(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        white_input = np.zeros(self.ft_weight.shape[1], dtype=np.float32)
        black_input = np.zeros(self.ft_weight.shape[1], dtype=np.float32)

        for f in white_features:
            if 0 <= f < len(white_input):
                white_input[f] = 1.0
        for f in black_features:
            if 0 <= f < len(black_input):
                black_input[f] = 1.0

        white_hidden = np.clip(np.dot(white_input, self.ft_weight.T) + self.ft_bias, 0, 1)
        black_hidden = np.clip(np.dot(black_input, self.ft_weight.T) + self.ft_bias, 0, 1)

        if stm:
            hidden = np.concatenate([white_hidden, black_hidden])
        else:
            hidden = np.concatenate([black_hidden, white_hidden])

        x = np.clip(np.dot(hidden, self.l1_weight.T) + self.l1_bias, 0, 1)
        x = np.clip(np.dot(x, self.l2_weight.T) + self.l2_bias, 0, 1)
        output = np.dot(x, self.l3_weight.T) + self.l3_bias

        return output[0]

    def evaluate_incremental(self, stm: bool) -> float:
        """PHASE 2: Uses Cython-accelerated evaluation when available.

        Supports L1 quantization modes based on L1_QUANTIZATION setting.
        """
        if self.white_accumulator is None or self.black_accumulator is None:
            raise RuntimeError("Accumulators not initialized.")

        # TODO: store accumulators in quantized form for additional speedup

        if self._quantization_mode == "INT8":
            return _cy_nnue_eval_int8(
                self.white_accumulator,
                self.black_accumulator,
                stm,
                self._l1_weight_q,
                self.l1_bias,
                self._l1_combined_scale,
                self.l2_weight,
                self.l2_bias,
                self.l3_weight,
                self.l3_bias,
                self._hidden_buf,
                self._hidden_buf_q,
                self._l1_buf,
                self._l2_buf,
                self._white_clipped,
                self._black_clipped
            )
        elif self._quantization_mode == "INT16":
            return _cy_nnue_eval_int16(
                self.white_accumulator,
                self.black_accumulator,
                stm,
                self._l1_weight_q,
                self.l1_bias,
                self._l1_combined_scale,
                self.l2_weight,
                self.l2_bias,
                self.l3_weight,
                self.l3_bias,
                self._hidden_buf,
                self._hidden_buf_q,
                self._l1_buf,
                self._l2_buf,
                self._white_clipped,
                self._black_clipped
            )
        else:
            # FP32 path (original)
            return _cy_nnue_eval(
                self.white_accumulator,
                self.black_accumulator,
                stm,
                self.l1_weight,
                self.l1_bias,
                self.l2_weight,
                self.l2_bias,
                self.l3_weight,
                self.l3_bias,
                self._hidden_buf,
                self._l1_buf,
                self._l2_buf,
                self._white_clipped,
                self._black_clipped
            )

    def refresh_accumulator(self, white_features: List[int], black_features: List[int]):
        self.white_accumulator = self.ft_bias.copy()
        self.black_accumulator = self.ft_bias.copy()

        valid_white = [f for f in white_features if 0 <= f < self._max_feature_idx]
        valid_black = [f for f in black_features if 0 <= f < self._max_feature_idx]

        if valid_white:
            self.white_accumulator += self.ft_weight[:, valid_white].sum(axis=1)
        if valid_black:
            self.black_accumulator += self.ft_weight[:, valid_black].sum(axis=1)

    def update_accumulator(self, added_features_white: Set[int], removed_features_white: Set[int],
                           added_features_black: Set[int], removed_features_black: Set[int]):
        """PHASE 2: Uses Cython-accelerated update when available."""
        if self.white_accumulator is None or self.black_accumulator is None:
            raise RuntimeError("Accumulators not initialized.")

        _cy_nnue_update(
            self.white_accumulator,
            self.black_accumulator,
            self.ft_weight,
            added_features_white,
            removed_features_white,
            added_features_black,
            removed_features_black,
            self._max_feature_idx
        )

    def evaluate_board(self, board: CachedBoard) -> float:
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)


class DNNInference:
    """
    PHASE 2 OPTIMIZED: Numpy-based inference engine for DNN.
    Uses Cython-accelerated operations when available.
    """

    def __init__(self, model: DNNNetwork):
        model.eval()
        with torch.no_grad():
            self.l1_weight = model.l1.weight.cpu().numpy().astype(np.float32)
            self.l1_bias = model.l1.bias.cpu().numpy().astype(np.float32)
            self.l2_weight = model.l2.weight.cpu().numpy().astype(np.float32)
            self.l2_bias = model.l2.bias.cpu().numpy().astype(np.float32)
            self.l3_weight = model.l3.weight.cpu().numpy().astype(np.float32)
            self.l3_bias = model.l3.bias.cpu().numpy().astype(np.float32)
            self.l4_weight = model.l4.weight.cpu().numpy().astype(np.float32)
            self.l4_bias = model.l4.bias.cpu().numpy().astype(np.float32)

        self.white_accumulator = None
        self.black_accumulator = None

        # Pre-allocate all buffers
        self._l2_buf = np.empty(self.l2_bias.shape[0], dtype=np.float32)
        self._l3_buf = np.empty(self.l3_bias.shape[0], dtype=np.float32)
        self._acc_clipped = np.empty(self.l1_bias.shape[0], dtype=np.float32)
        self._max_feature_idx = self.l1_weight.shape[1]

    def evaluate_full(self, features: List[int]) -> float:
        feature_input = np.zeros(DNN_INPUT_SIZE, dtype=np.float32)

        for f in features:
            if 0 <= f < len(feature_input):
                feature_input[f] = 1.0

        x = np.clip(np.dot(feature_input, self.l1_weight.T) + self.l1_bias, 0, 1)
        x = np.clip(np.dot(x, self.l2_weight.T) + self.l2_bias, 0, 1)
        x = np.clip(np.dot(x, self.l3_weight.T) + self.l3_bias, 0, 1)
        output = np.dot(x, self.l4_weight.T) + self.l4_bias

        return output[0]

    def evaluate_incremental(self, perspective: bool) -> float:
        """PHASE 2: Uses Cython-accelerated evaluation when available."""
        if perspective:
            if self.white_accumulator is None:
                raise RuntimeError("White accumulator not initialized")
            accumulator = self.white_accumulator
        else:
            if self.black_accumulator is None:
                raise RuntimeError("Black accumulator not initialized")
            accumulator = self.black_accumulator

        return _cy_dnn_eval(
            accumulator,
            self.l2_weight,
            self.l2_bias,
            self.l3_weight,
            self.l3_bias,
            self.l4_weight,
            self.l4_bias,
            self._l2_buf,
            self._l3_buf,
            self._acc_clipped
        )

    def refresh_accumulator(self, features: List[int], perspective: bool):
        accumulator = self.l1_bias.copy()

        valid_features = [f for f in features if 0 <= f < self._max_feature_idx]
        if valid_features:
            accumulator += self.l1_weight[:, valid_features].sum(axis=1)

        if perspective:
            self.white_accumulator = accumulator
        else:
            self.black_accumulator = accumulator

    def update_accumulator(self, added_features: Set[int], removed_features: Set[int], perspective: bool):
        """PHASE 2: Uses Cython-accelerated update when available."""
        if perspective:
            if self.white_accumulator is None:
                raise RuntimeError("White accumulator not initialized")
            accumulator = self.white_accumulator
        else:
            if self.black_accumulator is None:
                raise RuntimeError("Black accumulator not initialized")
            accumulator = self.black_accumulator

        _cy_dnn_update(
            accumulator,
            self.l1_weight,
            added_features,
            removed_features,
            self._max_feature_idx
        )

    def evaluate_board(self, board: CachedBoard) -> float:
        feat = DNNFeatures.board_to_features(board)
        return self.evaluate_full(feat)


def load_model(model_path: str, nn_type: str):
    """Load trained model from checkpoint file"""
    print(f"Loading {nn_type} model from {model_path}...")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        if nn_type == "NNUE":
            model = NNUENetwork(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
        elif nn_type == "DNN":
            model = DNNNetwork()
        else:
            raise ValueError(f"Unknown NN_TYPE: {nn_type}. Must be 'NNUE' or 'DNN'")

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  Checkpoint info: Epoch {checkpoint.get('epoch', 'unknown')}, "
                      f"Val loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        if nn_type == "NNUE":
            inference = NNUEInference(model)
        else:
            inference = DNNInference(model)

        print(f"✓ Model loaded successfully")
        return inference

    except FileNotFoundError:
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


