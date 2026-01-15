from typing import List, Tuple, Set, Dict, Optional

import chess
import numpy as np
import torch
from torch import nn as nn

NN_TYPE = "NNUE"

KING_SQUARES = 64
PIECE_SQUARES = 64
PIECE_TYPES = 5  # P, N, B, R, Q (no King)
COLORS = 2  # White, Black
NNUE_INPUT_SIZE = KING_SQUARES * PIECE_SQUARES * PIECE_TYPES * COLORS
NNUE_HIDDEN_SIZE = 256
DNN_INPUT_SIZE = 768  # 64 squares * 6 piece types * 2 colors
DNN_HIDDEN_LAYERS = [1024, 256, 32]
INPUT_SIZE = NNUE_INPUT_SIZE if NN_TYPE == "NNUE" else DNN_INPUT_SIZE
FIRST_HIDDEN_SIZE = NNUE_HIDDEN_SIZE if NN_TYPE == "NNUE" else DNN_HIDDEN_LAYERS[0]
OUTPUT_SIZE = 1
MODEL_PATH = "model/nnue.pt" if NN_TYPE == "NNUE" else "model/dnn.pt"
TANH_SCALE = 400
MAX_SCORE = 10_000
MATE_FACTOR = 100
MAX_MATE_DEPTH = 10
MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR

class NNUEFeatures:
    """Handles feature extraction for NNUE network"""

    @staticmethod
    def get_piece_index(piece_type: int, is_friendly_piece_color: bool) -> int:
        """Convert piece type and color to index (0-9)
        Note: This method uses the is_friendly_piece boolean as-is. The caller is
        responsible for flipping is_friendly_piece when extracting from black's perspective
        to achieve relative encoding (my pieces vs opponent pieces).
        """
        if piece_type == chess.KING:
            return -1
        type_idx = piece_type - 1
        color_idx = 1 if is_friendly_piece_color else 0
        return type_idx + color_idx * PIECE_TYPES

    @staticmethod
    def get_feature_index(king_sq: int, piece_sq: int, piece_type: int, is_friendly_piece: bool) -> int:
        """Calculate the feature index for (king_square, piece_square, piece_type, is_friendly_piece)"""
        piece_idx = NNUEFeatures.get_piece_index(piece_type, is_friendly_piece)
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
            is_friendly_piece = piece.color
            piece_type = piece.piece_type

            if not perspective:
                piece_square = NNUEFeatures.flip_square(piece_square)
                is_friendly_piece = not is_friendly_piece

            feature_idx = NNUEFeatures.get_feature_index(
                king_square, piece_square, piece_type, is_friendly_piece
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


class DNNFeatures:
    """Handles feature extraction for DNN network (768-dimensional one-hot encoding)"""

    @staticmethod
    def get_piece_index(piece_type: int, is_friendly_piece: bool) -> int:
        """Convert piece type and color to index (0-11) using relative colors"""
        # King=0, Queen=1, Rook=2, Bishop=3, Knight=4, Pawn=5
        # Friendly pieces: 0-5, Opponent pieces: 6-11
        type_map = {
            chess.KING: 0,
            chess.QUEEN: 1,
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4,
            chess.PAWN: 5
        }
        type_idx = type_map.get(piece_type, 0)
        color_offset = 0 if is_friendly_piece else 6
        return type_idx + color_offset

    @staticmethod
    def extract_features(board: chess.Board, perspective: bool) -> List[int]:
        """
        Extract 768-dimensional one-hot features from perspective of player to move.
        Returns list of active feature indices.

        Feature encoding: square * 12 + piece_index
        where piece_index is 0-11 (6 piece types × 2 colors)
        """
        features = []

        # If perspective is BLACK, we need to flip the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Adjust square if viewing from black's perspective
            if not perspective:
                # Flip square vertically (A1 <-> A8)
                rank = square // 8
                file = square % 8
                square = (7 - rank) * 8 + file

            # Get piece index (0-11)
            # Piece color should be relative to perspective:
            # True = friendly piece (side to move), False = opponent piece
            is_friendly_piece = (piece.color == chess.WHITE) == perspective
            piece_idx = DNNFeatures.get_piece_index(piece.piece_type, is_friendly_piece)
            # Calculate feature index: square * 12 + piece_index
            feature_idx = square * 12 + piece_idx
            features.append(feature_idx)

        return features

    @staticmethod
    def board_to_features(board: chess.Board) -> List[int]:
        """Extract features from perspective of side to move"""
        return DNNFeatures.extract_features(board, board.turn == chess.WHITE)


class NNUEIncrementalUpdater:
    """
    Efficiently maintains NNUE features with incremental updates and undo support.
    Uses a history stack to track changes, enabling O(k) pop() operations.

    For king moves (which require full recomputation), the entire feature set is saved
    for that perspective to enable efficient restoration.
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

        # History stack for efficient undo
        self.history_stack: List[Dict] = []

        # Store the last change for easy accumulator updates
        self.last_change: Optional[Dict] = None

    def _get_feature_for_perspective(self, perspective: bool, piece_sq: int,
                                     piece_type: int, is_friendly_piece: bool) -> int:
        """Get feature index for a piece from a given perspective"""
        if perspective:
            king_sq = self.white_king_sq
        else:
            king_sq = NNUEFeatures.flip_square(self.black_king_sq)
            piece_sq = NNUEFeatures.flip_square(piece_sq)
            is_friendly_piece = not is_friendly_piece

        return NNUEFeatures.get_feature_index(king_sq, piece_sq, piece_type, is_friendly_piece)

    def _remove_piece_features(self, square: int, piece_type: int, piece_color: bool,
                               change_record: Dict):
        """Remove features for a piece at the given square and record the change"""
        if piece_type == chess.KING:
            return

        # Remove from white's perspective
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat >= 0 and white_feat in self.white_features:
            self.white_features.discard(white_feat)
            change_record['white_removed'].add(white_feat)

        # Remove from black's perspective
        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat >= 0 and black_feat in self.black_features:
            self.black_features.discard(black_feat)
            change_record['black_removed'].add(black_feat)

    def _add_piece_features(self, square: int, piece_type: int, piece_color: bool,
                            change_record: Dict):
        """Add features for a piece at the given square and record the change"""
        if piece_type == chess.KING:
            return

        # Add to white's perspective
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat >= 0 and white_feat not in self.white_features:
            self.white_features.add(white_feat)
            change_record['white_added'].add(white_feat)

        # Add to black's perspective
        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat >= 0 and black_feat not in self.black_features:
            self.black_features.add(black_feat)
            change_record['black_added'].add(black_feat)

    def _recompute_perspective(self, perspective: bool):
        """Fully recompute features for one perspective (needed after king moves)"""
        features = set(NNUEFeatures.extract_features(self.board, perspective))
        if perspective:
            self.white_features = features
        else:
            self.black_features = features

    def push(self, move: chess.Move):
        """
        Update features after making a move and save changes for efficient undo.
        Must be called BEFORE board.push(move).
        """
        from_sq = move.from_square
        to_sq = move.to_square

        # Initialize change tracking for this move
        change_record = {
            'white_added': set(),
            'white_removed': set(),
            'black_added': set(),
            'black_removed': set(),
            'white_king_moved': False,
            'black_king_moved': False,
            'old_white_king_sq': self.white_king_sq,
            'old_black_king_sq': self.black_king_sq,
            'old_white_features': None,  # Will store full set if king moves
            'old_black_features': None  # Will store full set if king moves
        }

        piece = self.board.piece_at(from_sq)
        if piece is None:
            # Should not happen in valid games
            self.board.push(move)
            self.history_stack.append(change_record)
            self.last_change = change_record
            return

        moving_piece_type = piece.piece_type
        moving_piece_color = piece.color

        # Check if this is a king move
        is_white_king_move = (moving_piece_type == chess.KING and moving_piece_color == chess.WHITE)
        is_black_king_move = (moving_piece_type == chess.KING and moving_piece_color == chess.BLACK)

        # For king moves, save the entire feature set before recomputation
        if is_white_king_move:
            change_record['white_king_moved'] = True
            change_record['old_white_features'] = self.white_features.copy()
        if is_black_king_move:
            change_record['black_king_moved'] = True
            change_record['old_black_features'] = self.black_features.copy()

        # Check for capture
        captured_piece = self.board.piece_at(to_sq)
        is_en_passant = self.board.is_en_passant(move)

        # Handle en passant capture
        if is_en_passant:
            # The captured pawn is not on to_sq, it's on the en passant square
            ep_sq = to_sq + (-8 if moving_piece_color == chess.WHITE else 8)
            captured_piece = self.board.piece_at(ep_sq)
            if captured_piece:
                self._remove_piece_features(ep_sq, captured_piece.piece_type,
                                            captured_piece.color, change_record)

        # Remove captured piece features
        if captured_piece and not is_en_passant:
            self._remove_piece_features(to_sq, captured_piece.piece_type,
                                        captured_piece.color, change_record)

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

            # Remove rook from old square
            self._remove_piece_features(rook_from, chess.ROOK, moving_piece_color, change_record)

        # Remove moving piece from old square (if not king - king features handled by recompute)
        if not is_white_king_move and not is_black_king_move:
            self._remove_piece_features(from_sq, moving_piece_type, moving_piece_color, change_record)

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
            self._add_piece_features(to_sq, moving_piece_type, moving_piece_color, change_record)

        # Add rook to new square for castling
        if is_castling:
            self._add_piece_features(rook_to, chess.ROOK, moving_piece_color, change_record)

        # If king moved, recompute that perspective entirely
        # (all features depend on king square)
        if is_white_king_move:
            self._recompute_perspective(chess.WHITE)
        elif is_black_king_move:
            self._recompute_perspective(chess.BLACK)

        # Save the change record for efficient undo
        self.history_stack.append(change_record)
        self.last_change = change_record

    def pop(self) -> Dict:
        """
        Efficiently undo the last move using the history stack.

        For regular moves: O(k) complexity where k = number of features changed (typically 2-8).
        For king moves: O(n) complexity where n = total pieces, as we restore the saved feature set.

        This is still much more efficient than full recomputation from scratch.

        Returns:
            Dictionary with the changes that were reversed:
            - 'white_added': Features that were added to white (now removed)
            - 'white_removed': Features that were removed from white (now restored)
            - 'black_added': Features that were added to black (now removed)
            - 'black_removed': Features that were removed from black (now restored)
            - 'white_king_moved': Whether white king moved
            - 'black_king_moved': Whether black king moved
        """
        if not self.history_stack:
            raise ValueError("No moves to pop - history stack is empty")

        # Undo the board move
        self.board.pop()

        # Retrieve and remove the last change record
        change_record = self.history_stack.pop()

        # Restore king squares
        self.white_king_sq = change_record['old_white_king_sq']
        self.black_king_sq = change_record['old_black_king_sq']

        # Handle white perspective
        if change_record['white_king_moved']:
            # King moved: restore the saved feature set
            self.white_features = change_record['old_white_features']
        else:
            # Regular move: reverse the incremental changes
            self.white_features -= change_record['white_added']
            self.white_features |= change_record['white_removed']

        # Handle black perspective
        if change_record['black_king_moved']:
            # King moved: restore the saved feature set
            self.black_features = change_record['old_black_features']
        else:
            # Regular move: reverse the incremental changes
            self.black_features -= change_record['black_added']
            self.black_features |= change_record['black_removed']

        # Store as last change
        self.last_change = change_record

        return change_record

    def get_last_change(self) -> Optional[Dict]:
        """
        Get the changes from the last push() or pop() operation.
        Useful for updating accumulators.

        Returns:
            Dictionary with change information or None if no operation has been performed yet.
        """
        return self.last_change

    def get_features(self) -> Tuple[List[int], List[int]]:
        """Get current features as sorted lists"""
        return sorted(self.white_features), sorted(self.black_features)

    def get_features_unsorted(self) -> Tuple[List[int], List[int]]:
        """Get current features as lists (faster, no sorting)"""
        return list(self.white_features), list(self.black_features)

    def clear_history(self):
        """Clear the history stack to free memory (call after search completes)"""
        self.history_stack.clear()

    def history_size(self) -> int:
        """Get the number of moves in the history stack"""
        return len(self.history_stack)


class NNUENetwork(nn.Module):
    """PyTorch NNUE Network"""

    def __init__(self, input_size=INPUT_SIZE, hidden_size=FIRST_HIDDEN_SIZE):
        super(NNUENetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ft = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_features, black_features, stm):
        """
        Input: white_features (40960), black_features (40960), stm (1 or 0)
          ↓
        Feature Transform (shared weights): ft layer
          ↓
        w_hidden (256) and b_hidden (256)
          ↓
        Perspective Concatenation based on side-to-move
          ↓
        hidden (512) = [my_perspective, opponent_perspective]
          ↓
        l1: 512 → 32 (Clipped ReLU)
          ↓
        l2: 32 → 32 (Clipped ReLU)
          ↓
        l3: 32 → 1 (Linear)
          ↓
        Output: evaluation score
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
            # Positions where WHITE to move:
            hidden[white_to_move, :self.hidden_size] = w_hidden[white_to_move]  # My perspective (white's)
            hidden[white_to_move, self.hidden_size:] = b_hidden[white_to_move]  # Opponent perspective (black's)

            # Positions where BLACK to move:
            hidden[~white_to_move, :self.hidden_size] = b_hidden[~white_to_move]  # My perspective (black's)
            hidden[~white_to_move, self.hidden_size:] = w_hidden[~white_to_move]  # Opponent perspective (white's)
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
        return x  # Linear output (no activation)


class DNNNetwork(nn.Module):
    """PyTorch DNN Network for chess position evaluation"""

    def __init__(self, input_size=DNN_INPUT_SIZE, hidden_layers=None):
        super(DNNNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # Build layers: input -> hidden[0] -> hidden[1] -> hidden[2] -> output
        self.l1 = nn.Linear(input_size, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, features):
        """
        features: dense tensor of shape (batch_size, 768)
        """
        x = torch.clamp(self.l1(features), 0, 1)  # Clamped ReLU
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.clamp(self.l3(x), 0, 1)
        x = self.l4(x)
        return x  # Linear output (no activation)


class DNNIncrementalUpdater:
    # TODO remove board from class. Pass before/after_push/pop_board
    """
    Incrementally maintains DNN features for both perspectives with efficient undo support.
    Uses a history stack to track changes, enabling O(k) pop() operations where k = pieces affected.
    Provides methods to retrieve change information for accumulator updates.
    """

    def __init__(self, board: chess.Board):
        """Initialize with a board position"""
        self.board = board.copy()
        self.white_features: Set[int] = set(DNNFeatures.extract_features(board, chess.WHITE))
        self.black_features: Set[int] = set(DNNFeatures.extract_features(board, chess.BLACK))

        # History stack for efficient undo
        self.history_stack: List[Dict[str, Set[int]]] = []

        # Store the last change for easy accumulator updates
        self.last_change: Optional[Dict[str, Set[int]]] = None

    def _get_feature_for_perspective(self, perspective: bool, square: int,
                                     piece_type: int, piece_color: bool) -> int:
        """Get feature index for a piece from a given perspective"""
        if not perspective:
            rank = square // 8
            file = square % 8
            square = (7 - rank) * 8 + file

        is_friendly_piece = (piece_color == chess.WHITE) == perspective
        piece_idx = DNNFeatures.get_piece_index(piece_type, is_friendly_piece)
        return square * 12 + piece_idx

    def _remove_piece_features(self, square: int, piece_type: int, piece_color: bool,
                               change_record: Dict[str, Set[int]]):
        """Remove features for a piece and record the change"""
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
        """Add features for a piece and record the change"""
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        if white_feat not in self.white_features:
            self.white_features.add(white_feat)
            change_record['white_added'].add(white_feat)

        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        if black_feat not in self.black_features:
            self.black_features.add(black_feat)
            change_record['black_added'].add(black_feat)

    def push(self, move: chess.Move):
        """Update features after making a move and save changes for efficient undo"""
        from_sq = move.from_square
        to_sq = move.to_square

        # Initialize change tracking for this move
        change_record = {
            'white_added': set(),
            'white_removed': set(),
            'black_added': set(),
            'black_removed': set()
        }

        piece = self.board.piece_at(from_sq)
        if piece is None:
            self.board.push(move)
            self.history_stack.append(change_record)
            self.last_change = change_record
            return

        moving_piece_type = piece.piece_type
        moving_piece_color = piece.color

        captured_piece = self.board.piece_at(to_sq)
        is_en_passant = self.board.is_en_passant(move)

        # Handle en passant capture
        if is_en_passant:
            ep_sq = to_sq + (-8 if moving_piece_color == chess.WHITE else 8)
            captured_piece = self.board.piece_at(ep_sq)
            if captured_piece:
                self._remove_piece_features(ep_sq, captured_piece.piece_type,
                                            captured_piece.color, change_record)

        # Handle regular capture
        if captured_piece and not is_en_passant:
            self._remove_piece_features(to_sq, captured_piece.piece_type,
                                        captured_piece.color, change_record)

        # Handle castling - move the rook
        is_castling = self.board.is_castling(move)
        if is_castling:
            if to_sq > from_sq:  # Kingside
                rook_from = chess.H1 if moving_piece_color == chess.WHITE else chess.H8
                rook_to = chess.F1 if moving_piece_color == chess.WHITE else chess.F8
            else:  # Queenside
                rook_from = chess.A1 if moving_piece_color == chess.WHITE else chess.A8
                rook_to = chess.D1 if moving_piece_color == chess.WHITE else chess.D8

            self._remove_piece_features(rook_from, chess.ROOK, moving_piece_color, change_record)

        # Remove moving piece from old square
        self._remove_piece_features(from_sq, moving_piece_type, moving_piece_color, change_record)

        # Make the move
        self.board.push(move)

        # Handle promotion
        if move.promotion:
            moving_piece_type = move.promotion

        # Add moving piece to new square
        self._add_piece_features(to_sq, moving_piece_type, moving_piece_color, change_record)

        # Add rook to new square for castling
        if is_castling:
            self._add_piece_features(rook_to, chess.ROOK, moving_piece_color, change_record)

        # Save the change record for efficient undo
        self.history_stack.append(change_record)
        self.last_change = change_record

    def pop(self) -> Dict[str, Set[int]]:
        """
        Efficiently undo the last move using the history stack.
        O(k) complexity where k = number of features that changed (typically 2-8).

        Returns:
            Dictionary with the changes that were reversed:
            - 'white_added': Features that were added to white (now removed)
            - 'white_removed': Features that were removed from white (now restored)
            - 'black_added': Features that were added to black (now removed)
            - 'black_removed': Features that were removed from black (now restored)
        """
        if not self.history_stack:
            raise ValueError("No moves to pop - history stack is empty")

        # Undo the board move
        self.board.pop()

        # Retrieve and remove the last change record
        change_record = self.history_stack.pop()

        # Reverse the changes:
        # - Features that were added must be removed
        # - Features that were removed must be added back
        self.white_features -= change_record['white_added']
        self.white_features |= change_record['white_removed']
        self.black_features -= change_record['black_added']
        self.black_features |= change_record['black_removed']

        # Store as last change (but reversed)
        self.last_change = change_record

        return change_record

    def get_last_change(self) -> Optional[Dict[str, Set[int]]]:
        """
        Get the changes from the last push() or pop() operation.
        Useful for updating accumulators.

        Returns:
            Dictionary with 'white_added', 'white_removed', 'black_added', 'black_removed'
            or None if no operation has been performed yet.
        """
        return self.last_change

    def get_features(self) -> List[int]:
        """Get features for current side to move"""
        if self.board.turn == chess.WHITE:
            return list(self.white_features)
        else:
            return list(self.black_features)

    def get_features_both(self) -> Tuple[List[int], List[int]]:
        """Get features for both perspectives"""
        return list(self.white_features), list(self.black_features)

    def clear_history(self):
        """Clear the history stack to free memory (call after search completes)"""
        self.history_stack.clear()

    def history_size(self) -> int:
        """Get the number of moves in the history stack"""
        return len(self.history_stack)


class NNUEInference:
    """Numpy-based inference engine for NNUE with incremental evaluation support"""

    def __init__(self, model: NNUENetwork):
        """Extract weights from PyTorch model to numpy arrays"""
        model.eval()
        with torch.no_grad():
            self.ft_weight = model.ft.weight.cpu().numpy()
            self.ft_bias = model.ft.bias.cpu().numpy()
            self.l1_weight = model.l1.weight.cpu().numpy()
            self.l1_bias = model.l1.bias.cpu().numpy()
            self.l2_weight = model.l2.weight.cpu().numpy()
            self.l2_bias = model.l2.bias.cpu().numpy()
            self.l3_weight = model.l3.weight.cpu().numpy()
            self.l3_bias = model.l3.bias.cpu().numpy()

        self.hidden_size = model.hidden_size

        # For incremental evaluation (accumulator-based)
        self.white_accumulator = None
        self.black_accumulator = None

    @staticmethod
    def clipped_relu(x):
        """Clipped ReLU activation [0, 1]"""
        return np.clip(x, 0, 1)

    def evaluate_full(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        """
        Full evaluation using matrix multiplication.

        Args:
            white_features: Active feature indices for white's perspective
            black_features: Active feature indices for black's perspective
            stm: True if white to move, False if black to move

        Returns:
            Evaluation score (linear output, approximately in [-1, 1])
        """
        # Create sparse input vectors
        white_input = np.zeros(self.ft_weight.shape[1], dtype=np.float32)
        black_input = np.zeros(self.ft_weight.shape[1], dtype=np.float32)

        for f in white_features:
            if 0 <= f < len(white_input):
                white_input[f] = 1.0
        for f in black_features:
            if 0 <= f < len(black_input):
                black_input[f] = 1.0

        # Feature transform - MATRIX MULTIPLICATION
        white_hidden = self.clipped_relu(np.dot(white_input, self.ft_weight.T) + self.ft_bias)
        black_hidden = self.clipped_relu(np.dot(black_input, self.ft_weight.T) + self.ft_bias)

        # Perspective concatenation
        if stm:
            hidden = np.concatenate([white_hidden, black_hidden])
        else:
            hidden = np.concatenate([black_hidden, white_hidden])

        # Further layers
        x = self.clipped_relu(np.dot(hidden, self.l1_weight.T) + self.l1_bias)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        output = np.dot(x, self.l3_weight.T) + self.l3_bias

        return output[0]

    def evaluate_incremental(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        """
        Incremental evaluation using accumulators (add/subtract instead of matrix multiply).
        Must call _refresh_accumulator() before first use!
        """
        if self.white_accumulator is None or self.black_accumulator is None:
            raise RuntimeError("Accumulators not initialized. Call _refresh_accumulator() first.")

        # Apply clipped relu to accumulators (no matrix multiply!)
        white_hidden = self.clipped_relu(self.white_accumulator)
        black_hidden = self.clipped_relu(self.black_accumulator)

        # Perspective concatenation
        if stm:
            hidden = np.concatenate([white_hidden, black_hidden])
        else:
            hidden = np.concatenate([black_hidden, white_hidden])

        # Further layers (same as full evaluation)
        x = self.clipped_relu(np.dot(hidden, self.l1_weight.T) + self.l1_bias)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        output = np.dot(x, self.l3_weight.T) + self.l3_bias

        return output[0]

    def _refresh_accumulator(self, white_features: List[int], black_features: List[int]):
        """Refresh accumulators from features"""
        self.white_accumulator = self.ft_bias.copy()
        self.black_accumulator = self.ft_bias.copy()

        for f in white_features:
            if 0 <= f < self.ft_weight.shape[1]:
                self.white_accumulator += self.ft_weight[:, f]

        for f in black_features:
            if 0 <= f < self.ft_weight.shape[1]:
                self.black_accumulator += self.ft_weight[:, f]

    def update_accumulator(self, added_features_white: Set[int], removed_features_white: Set[int],
                           added_features_black: Set[int], removed_features_black: Set[int]):
        """Update accumulators incrementally"""
        if self.white_accumulator is None or self.black_accumulator is None:
            raise RuntimeError("Accumulators not initialized. Call _refresh_accumulator() first.")

        for f in added_features_white:
            if 0 <= f < self.ft_weight.shape[1]:
                self.white_accumulator += self.ft_weight[:, f]
        for f in removed_features_white:
            if 0 <= f < self.ft_weight.shape[1]:
                self.white_accumulator -= self.ft_weight[:, f]

        for f in added_features_black:
            if 0 <= f < self.ft_weight.shape[1]:
                self.black_accumulator += self.ft_weight[:, f]
        for f in removed_features_black:
            if 0 <= f < self.ft_weight.shape[1]:
                self.black_accumulator -= self.ft_weight[:, f]

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board position (uses full evaluation)"""
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)


class DNNInference:
    """Numpy-based inference engine for DNN with incremental evaluation support"""

    def __init__(self, model: DNNNetwork):
        """Extract weights from PyTorch model to numpy arrays"""
        model.eval()
        with torch.no_grad():
            self.l1_weight = model.l1.weight.cpu().numpy()
            self.l1_bias = model.l1.bias.cpu().numpy()
            self.l2_weight = model.l2.weight.cpu().numpy()
            self.l2_bias = model.l2.bias.cpu().numpy()
            self.l3_weight = model.l3.weight.cpu().numpy()
            self.l3_bias = model.l3.bias.cpu().numpy()
            self.l4_weight = model.l4.weight.cpu().numpy()
            self.l4_bias = model.l4.bias.cpu().numpy()

        # For incremental evaluation (accumulator-based)
        self.white_accumulator = None
        self.black_accumulator = None

    @staticmethod
    def clipped_relu(x):
        """Clipped ReLU activation [0, 1]"""
        return np.clip(x, 0, 1)

    def evaluate_full(self, features: List[int]) -> float:
        """Full evaluation using matrix multiplication"""
        feature_input = np.zeros(DNN_INPUT_SIZE, dtype=np.float32)

        for f in features:
            if 0 <= f < len(feature_input):
                feature_input[f] = 1.0

        # Forward pass - MATRIX MULTIPLICATION
        x = self.clipped_relu(np.dot(feature_input, self.l1_weight.T) + self.l1_bias)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        x = self.clipped_relu(np.dot(x, self.l3_weight.T) + self.l3_bias)
        output = np.dot(x, self.l4_weight.T) + self.l4_bias

        return output[0]

    def evaluate_incremental(self, features: List[int], perspective: bool) -> float:
        """Incremental evaluation using accumulator"""
        if perspective:
            if self.white_accumulator is None:
                raise RuntimeError("White accumulator not initialized")
            accumulator = self.white_accumulator
        else:
            if self.black_accumulator is None:
                raise RuntimeError("Black accumulator not initialized")
            accumulator = self.black_accumulator

        x = self.clipped_relu(accumulator)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        x = self.clipped_relu(np.dot(x, self.l3_weight.T) + self.l3_bias)
        output = np.dot(x, self.l4_weight.T) + self.l4_bias

        return output[0]

    def _refresh_accumulator(self, features: List[int], perspective: bool):
        """Refresh accumulator from features"""
        accumulator = self.l1_bias.copy()

        for f in features:
            if 0 <= f < self.l1_weight.shape[1]:
                accumulator += self.l1_weight[:, f]

        if perspective:
            self.white_accumulator = accumulator
        else:
            self.black_accumulator = accumulator

    def update_accumulator(self, added_features: Set[int], removed_features: Set[int], perspective: bool):
        """Update accumulator incrementally"""
        if perspective:
            if self.white_accumulator is None:
                raise RuntimeError("White accumulator not initialized")
            accumulator = self.white_accumulator
        else:
            if self.black_accumulator is None:
                raise RuntimeError("Black accumulator not initialized")
            accumulator = self.black_accumulator

        for f in added_features:
            if 0 <= f < self.l1_weight.shape[1]:
                accumulator += self.l1_weight[:, f]
        for f in removed_features:
            if 0 <= f < self.l1_weight.shape[1]:
                accumulator -= self.l1_weight[:, f]

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board position (uses full evaluation)"""
        feat = DNNFeatures.board_to_features(board)
        return self.evaluate_full(feat)


