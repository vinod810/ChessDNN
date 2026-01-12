"""
Chess Neural Network Evaluation Script
Loads trained NNUE or DNN models and evaluates positions from FEN strings.
Uses numpy for fast inference (avoiding PyTorch overhead).

Test Modes:
- Interactive-FEN: Interactive FEN evaluation (default)
- Incremental-vs-Full: Performance comparison
- Accumulator-Correctness: Verify incremental == full evaluation
- Eval-accuracy: Test prediction accuracy against ground truth CSV
"""
import os

import numpy as np
import chess
import torch
import sys
import time
from typing import List, Set, Tuple
import chess
from typing import Set, List, Tuple, Dict
import chess
from typing import Set, List, Tuple, Dict, Optional
# Import required modules for PGN processing
import glob
import zstandard as zstd
import io
import chess.pgn
import os

# Import configuration and classes from train_nn
try:
    from nn_train import (
        MODEL_PATH, TANH_SCALE,
        NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE,
        DNN_INPUT_SIZE, DNN_HIDDEN_LAYERS,
        NNUEFeatures, DNNFeatures,
        NNUENetwork, DNNNetwork,
        NNUEIncrementalUpdater
    )
except ImportError:
    print("ERROR: Could not import from nn_train.py")
    print("Make sure nn_train.py (or train_nn_fixed.py) is in the same directory")
    sys.exit(1)

# Configuration
valid_test_types_dict = {0: "Interactive-FEN", 1: "Incremental-vs-Full", 2: "Accumulator-Correctness",
                         3: "Eval-Accuracy"}
valid_test_types = list(valid_test_types_dict.values())
test_type = None
nn_type = None


class DNNIncrementalUpdater:
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


def load_model(model_path: str, nn_type: str):
    """Load trained model from checkpoint file"""
    print(f"Loading {nn_type} model from {model_path}...")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        if nn_type == "NNUE":
            model = NNUENetwork()
        elif nn_type == "DNN":
            model = DNNNetwork()
        else:
            raise ValueError(f"Unknown NN_TYPE: {nn_type}. Must be 'NNUE' or 'DNN'")

        # Handle different checkpoint formats
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


def output_to_centipawns(output: float) -> float:
    """Convert linear network output to centipawns"""
    output = np.clip(output, -0.99, 0.99)
    centipawns = np.arctanh(output) * TANH_SCALE
    return centipawns


def evaluate_fen(fen: str, inference) -> dict:
    """Evaluate a position from FEN string"""
    try:
        board = chess.Board(fen)
        output = inference.evaluate_board(board)
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
    """Pretty print evaluation results"""
    if not result['success']:
        print(f"❌ {result['error']}")
        return

    print("─" * 60)
    print(f"FEN: {result['fen']}")
    print(f"Side to move: {result['side_to_move']}")
    print(f"Network output: {result['output']:.6f}")
    print(f"Evaluation: {result['centipawns']:+.2f} centipawns")

    pawns = result['centipawns'] / 100.0
    if abs(pawns) < 0.1:
        assessment = "≈ Equal position"
    elif pawns > 3:
        assessment = f"Decisive advantage for {result['side_to_move']}"
    elif pawns > 1:
        assessment = f"Clear advantage for {result['side_to_move']}"
    elif pawns > 0.3:
        assessment = f"Slight advantage for {result['side_to_move']}"
    elif pawns < -3:
        opponent = "Black" if result['side_to_move'] == "White" else "White"
        assessment = f"Decisive advantage for {opponent}"
    elif pawns < -1:
        opponent = "Black" if result['side_to_move'] == "White" else "White"
        assessment = f"Clear advantage for {opponent}"
    elif pawns < -0.3:
        opponent = "Black" if result['side_to_move'] == "White" else "White"
        assessment = f"Slight advantage for {opponent}"
    else:
        assessment = "≈ Equal position"

    print(f"Assessment: {assessment}")
    print("─" * 60)


def print_help():
    """Print help information"""
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


def test_accumulator_correctness_nnue(inference: NNUEInference):
    """Test accumulator correctness for NNUE using specific game"""
    print("\n" + "=" * 70)
    print("NNUE Accumulator Correctness Test")
    print("=" * 70)

    # Includes captures, en-passant, castlings, king moves and promotion.
    moves_san = ["d4", "e5", "dxe5", "f5", "exf6", "Nh6", "Bf4", "Bd6", "Nc3", "O-O", "Qd3", "Nc6", "O-O-O",
                 "a6", "fxg7", "Rb8", "gxf8=N", "Kxf8", "Kb1", "Bxf4"]

    # Play the game
    board = chess.Board()
    updater = NNUEIncrementalUpdater(board)

    # Initialize accumulator
    white_feat, black_feat = updater.get_features_unsorted()
    inference._refresh_accumulator(white_feat, black_feat)

    move_count = 0
    for move_san in moves_san:
        move = updater.board.parse_san(move_san)

        # Track old features
        old_white = set(updater.white_features)
        old_black = set(updater.black_features)

        # Push move
        updater.push(move)

        # Update accumulator
        new_white = set(updater.white_features)
        new_black = set(updater.black_features)
        inference.update_accumulator(
            new_white - old_white,
            old_white - new_white,
            new_black - old_black,
            old_black - new_black
        )

        move_count += 1

        # Get current features
        white_feat, black_feat = updater.get_features_unsorted()
        stm = updater.board.turn == chess.WHITE

        # Evaluate with incremental (using accumulator)
        eval_incremental = inference.evaluate_incremental(white_feat, black_feat, stm)

        # Evaluate with full (matrix multiplication from scratch)
        white_feat_full, black_feat_full = NNUEFeatures.board_to_features(updater.board)
        eval_full = inference.evaluate_full(white_feat_full, black_feat_full, stm)

        # Compare
        diff = abs(eval_incremental - eval_full)

        # Test 1:
        print("\n" + "─" * 70)
        print(f"Test 1: After move {move_count} ({move_san})")
        print("─" * 70)

        print(f"Position: {updater.board.fen()}")
        print(f"Incremental eval: {eval_incremental:.10f} ({output_to_centipawns(eval_incremental):+.2f} cp)")
        print(f"Full eval:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:       {diff:.10e}")

        if diff < 1e-6:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")

        # Test 2: Pop 8 times and compare again
        print("\n" + "─" * 70)
        print("Test 2: After popping 8 moves")
        print("─" * 70)

    print("Popping 8 moves...")
    for i in range(4):
        # Pop and update accumulator incrementally
        change_record = updater.pop()

        # Update accumulator to reflect the reversed changes
        inference.update_accumulator(
            change_record['white_removed'],  # Add back what was removed
            change_record['white_added'],  # Remove what was added
            change_record['black_removed'],  # Add back what was removed
            change_record['black_added']  # Remove what was added
        )
        stm = updater.board.turn == chess.WHITE

        # Evaluate with incremental
        eval_incremental = inference.evaluate_incremental(white_feat, black_feat, stm)

        # Evaluate with full
        eval_full = inference.evaluate_full(white_feat, black_feat, stm)

        # Compare
        diff = abs(eval_incremental - eval_full)

        print("\n" + "─" * 70)
        print(f"Test 1: After pop {i + 1}")
        print("─" * 70)

        print(f"Position: {updater.board.fen()}")
        print(f"Incremental eval: {eval_incremental:.10f} ({output_to_centipawns(eval_incremental):+.2f} cp)")
        print(f"Full eval:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:       {diff:.10e}")

        if diff < 1e-6:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")

        print("\n" + "=" * 70)
        print("Accumulator correctness test complete!")
        print("=" * 70)


def test_accumulator_correctness_dnn(inference: DNNInference):
    """Test accumulator correctness for DNN using specific game"""
    print("\n" + "=" * 70)
    print("DNN Accumulator Correctness Test")
    print("=" * 70)

    moves_san = ["d4", "e5", "dxe5", "f5", "exf6", "Nh6", "Bf4", "Bd6", "Nc3", "O-O", "Qd3", "Nc6", "O-O-O",
                 "a6", "fxg7", "Rb8", "gxf8=N", "Kxf8", "Kb1", "Bxf4"]

    # Play the game
    board = chess.Board()
    updater = DNNIncrementalUpdater(board)

    # Initialize accumulator
    white_feat, black_feat = updater.get_features_both()
    inference._refresh_accumulator(white_feat, True)
    inference._refresh_accumulator(black_feat, False)

    move_count = 0
    for move_san in moves_san:
        move = updater.board.parse_san(move_san)

        # FIX: Track old features for BOTH perspectives (not just current side)
        old_white = set(updater.white_features)
        old_black = set(updater.black_features)

        # Push move
        updater.push(move)

        # FIX: Get new features for BOTH perspectives
        new_white = set(updater.white_features)
        new_black = set(updater.black_features)

        # FIX: Update BOTH accumulators with correct changes
        # DNN's update_accumulator updates one perspective at a time
        inference.update_accumulator(
            new_white - old_white,  # white features added
            old_white - new_white,  # white features removed
            True  # white perspective
        )
        inference.update_accumulator(
            new_black - old_black,  # black features added
            old_black - new_black,  # black features removed
            False  # black perspective
        )

        move_count += 1

        # Get current features
        features = updater.get_features()
        perspective = updater.board.turn == chess.WHITE

        # Evaluate with incremental
        eval_incremental = inference.evaluate_incremental(features, perspective)

        # Evaluate with full
        features_full = DNNFeatures.board_to_features(updater.board)
        eval_full = inference.evaluate_full(features_full)

        # Compare
        diff = abs(eval_incremental - eval_full)

        # Test 1:
        print("\n" + "─" * 70)
        print(f"Test 1: After move {move_count} ({move_san})")
        print("─" * 70)

        print(f"Position: {updater.board.fen()}")
        print(f"Incremental eval: {eval_incremental:.10f} ({output_to_centipawns(eval_incremental):+.2f} cp)")
        print(f"Full eval:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:       {diff:.10e}")

        if diff < 1e-6:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")

    print("Popping 4 moves...")
    for i in range(4):
        # Pop and update accumulator incrementally
        change_record = updater.pop()

        # Update accumulator to reflect the reversed changes
        # When we pop, we reverse: what was added gets removed, what was removed gets added back
        inference.update_accumulator(
            change_record['white_removed'],  # Add back what was removed
            change_record['white_added'],  # Remove what was added
            True  # white perspective
        )
        inference.update_accumulator(
            change_record['black_removed'],  # Add back what was removed
            change_record['black_added'],  # Remove what was added
            False  # black perspective
        )

        features = updater.get_features()
        perspective = updater.board.turn == chess.WHITE

        # Evaluate with incremental
        eval_incremental = inference.evaluate_incremental(features, perspective)

        # Evaluate with full
        features_full = DNNFeatures.board_to_features(updater.board)
        eval_full = inference.evaluate_full(features_full)

        # Compare
        diff = abs(eval_incremental - eval_full)

        print("\n" + "─" * 70)
        print(f"Test 1: After pop {i + 1}")
        print("─" * 70)

        print(f"Position: {updater.board.fen()}")
        print(f"Incremental eval: {eval_incremental:.10f} ({output_to_centipawns(eval_incremental):+.2f} cp)")
        print(f"Full eval:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:       {diff:.10e}")

        if diff < 1e-6:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")

        print("\n" + "=" * 70)
        print("Accumulator correctness test complete!")
        print("=" * 70)


def test_eval_accuracy(inference, nn_type: str):
    """
    Test evaluation accuracy against ground truth from PGN file.
    Computes MSE of tanh(cp/400) between predicted and true scores.
    Extracts 1000 positions from annotated PGN games (matching ProcessGameWithValidation logic).
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Evaluation Accuracy Test")
    print("=" * 70)

    # Find PGN files
    pgn_dir = "pgn"
    pgn_files = glob.glob(f"{pgn_dir}/*.pgn.zst")

    if not pgn_files:
        print(f"\n❌ ERROR: No PGN files found in {pgn_dir}/")
        print("Please ensure .pgn.zst files exist in the pgn directory")
        return

    # Use the first PGN file
    pgn_file = pgn_files[0]
    print(f"\nReading positions from: {pgn_file}")

    # Configuration (matching ProcessGameWithValidation logic)
    MAX_PLYS_PER_GAME = 200
    OPENING_PLYS = 0
    TARGET_POSITIONS = 100_000

    # Import constants from training module
    from nn_train import TANH_SCALE, MAX_SCORE, MATE_FACTOR, MAX_MATE_DEPTH, MAX_NON_MATE_SCORE

    positions = []
    games_processed = 0

    try:
        print("Extracting positions from games...")

        with open(pgn_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                while len(positions) < TARGET_POSITIONS:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break  # End of file

                    games_processed += 1

                    # Skip variant games
                    if any("Variant" in key for key in game.headers.keys()):
                        continue

                    board = game.board()
                    move_count = 0
                    game_positions = []

                    # Process game mainline
                    for node in game.mainline():
                        move_count += 1
                        current_move = node.move

                        # Check if this move is a capture BEFORE pushing it
                        was_last_move_capture = board.is_capture(current_move)

                        if len(game_positions) >= MAX_PLYS_PER_GAME:
                            break

                        # Skip opening moves
                        if move_count <= OPENING_PLYS:
                            board.push(current_move)
                            continue

                        # Push the move
                        board.push(current_move)

                        # Skip game-over positions
                        if board.is_game_over():
                            continue

                        # Skip if side to move is in check
                        if board.is_check():
                            continue

                        # Skip if this move was a capture (position after capture is tactically unstable)
                        if was_last_move_capture:
                            continue

                        # Get evaluation
                        ev = node.eval()
                        if ev is None:
                            continue

                        # Convert evaluation to score (matching ProcessGameWithValidation logic)
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

                        # Store position data
                        # Score is from white's perspective, adjust for side to move
                        fen = board.fen()
                        true_cp = score_cp if board.turn == chess.WHITE else -score_cp

                        game_positions.append({
                            'fen': fen,
                            'true_cp': true_cp
                        })

                    # Add positions from this game to our collection
                    positions.extend(game_positions)

                    if games_processed % int(TARGET_POSITIONS / 10 / 10) == 0:
                        print(f"  Processed {games_processed} games, collected {len(positions)} positions...")

                    if len(positions) >= TARGET_POSITIONS:
                        break

        # Trim to exactly TARGET_POSITIONS
        positions = positions[:TARGET_POSITIONS]

        if not positions:
            print("\n❌ ERROR: No valid positions found in PGN file")
            print("Please ensure the PGN file contains games with engine evaluations")
            return

        print(f"✓ Loaded {len(positions)} positions from {games_processed} games")

    except Exception as e:
        print(f"❌ ERROR reading PGN: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate all positions
    print("\nEvaluating positions...")

    true_tanh_values = []
    pred_tanh_values = []
    errors = []

    for i, pos in enumerate(positions):
        if (i + 1) % int(TARGET_POSITIONS / 10) == 0:
            print(f"  Progress: {i + 1}/{len(positions)}")

        try:
            # Get predicted value
            board = chess.Board(pos['fen'])

            # Get true tanh value
            true_cp = pos['true_cp']
            true_tanh = np.tanh(true_cp / TANH_SCALE)

            # Use full evaluation for accuracy test
            if nn_type == "NNUE":
                white_feat, black_feat = NNUEFeatures.board_to_features(board)
                pred_output = inference.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)
            else:  # DNN
                features = DNNFeatures.board_to_features(board)
                pred_output = inference.evaluate_full(features)

            # pred_output is already in tanh space (network outputs ~tanh(cp/400))
            pred_tanh = pred_output

            true_tanh_values.append(true_tanh)
            pred_tanh_values.append(pred_tanh)

        except Exception as e:
            errors.append((i, pos['fen'], str(e)))

    if errors:
        print(f"\n⚠ {len(errors)} positions failed to evaluate:")
        for idx, fen, error in errors[:5]:  # Show first 5 errors
            print(f"  Position {idx}: {error[:60]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Compute MSE
    true_tanh_values = np.array(true_tanh_values)
    pred_tanh_values = np.array(pred_tanh_values)

    mse = np.mean((true_tanh_values - pred_tanh_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_tanh_values - pred_tanh_values))

    # Also compute MSE in centipawn space for reference
    true_cp_values = true_tanh_values * TANH_SCALE
    pred_cp_values = np.arctanh(np.clip(pred_tanh_values, -0.99, 0.99)) * TANH_SCALE
    mse_cp = np.mean((true_cp_values - pred_cp_values) ** 2)
    rmse_cp = np.sqrt(mse_cp)
    mae_cp = np.mean(np.abs(true_cp_values - pred_cp_values))

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
    print("Centipawn Space (for reference):")
    print(f"  MSE:  {mse_cp:.2f}")
    print(f"  RMSE: {rmse_cp:.2f} cp")
    print(f"  MAE:  {mae_cp:.2f} cp")

    # Show some example predictions
    print("\n" + "─" * 70)
    print("Sample predictions (first 20):")
    print("─" * 70)
    for i in range(min(20, len(positions))):
        true_cp = positions[i]['true_cp']
        pred_cp = pred_cp_values[i]
        diff = pred_cp - true_cp
        print(f"{i + 1:2d}. True: {true_cp:+7.1f} cp | Pred: {pred_cp:+7.1f} cp | Diff: {diff:+7.1f} cp")

    print("\n" + "=" * 70)
    print("Evaluation accuracy test complete!")
    print("=" * 70)

    # Evaluate all positions
    print("\nEvaluating positions...")


def performance_test_nnue(inference: NNUEInference):
    """Run performance comparison for NNUE: Full vs Incremental evaluation"""
    print("\n" + "=" * 70)
    print("NNUE Performance Test: Full vs Incremental Evaluation")
    print("=" * 70)

    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # Test 1: Full evaluation
    print("\n1. Full Evaluation - Matrix Multiplication (1000 iterations)")

    start_time = time.time()
    for _ in range(1000):
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        output = inference.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)
    full_time = time.time() - start_time

    print(f"   Time: {full_time:.4f} seconds")
    print(f"   Avg per evaluation: {full_time / 1000 * 1000:.3f} ms")

    # Test 2: Incremental evaluation
    print("\n2. Incremental Evaluation - Accumulator Add/Subtract (500 push/pop cycles)")

    updater = NNUEIncrementalUpdater(board)
    white_feat, black_feat = updater.get_features_unsorted()
    inference._refresh_accumulator(white_feat, black_feat)

    start_time = time.time()
    for _ in range(500):
        old_white = set(updater.white_features)
        old_black = set(updater.black_features)

        updater.push(move)

        new_white = set(updater.white_features)
        new_black = set(updater.black_features)

        inference.update_accumulator(
            new_white - old_white, old_white - new_white,
            new_black - old_black, old_black - new_black
        )

        output = inference.evaluate_incremental(
            list(new_white), list(new_black), updater.board.turn == chess.WHITE
        )

        updater.board.pop()
        updater.white_features = old_white
        updater.black_features = old_black

        inference.update_accumulator(
            old_white - new_white, new_white - old_white,
            old_black - new_black, new_black - old_black
        )

        output = inference.evaluate_incremental(
            list(old_white), list(old_black), board.turn == chess.WHITE
        )

    incremental_time = time.time() - start_time

    print(f"   Time: {incremental_time:.4f} seconds")
    print(f"   Avg per push/pop cycle: {incremental_time / 500 * 1000:.3f} ms")

    print("\n" + "─" * 70)
    print("Results:")
    print(f"  Full (matrix multiply):     {full_time / 1000 * 1000:.3f} ms per evaluation")
    print(f"  Incremental (add/subtract): {incremental_time / 500 * 1000:.3f} ms per cycle")
    print(f"  Speedup: {full_time / (incremental_time) * 500 / 1000:.2f}x")
    print("\nKey: Incremental uses accumulator (add/subtract weight vectors)")
    print("instead of full matrix multiplication for the first layer.")
    print("=" * 70)


def performance_test_dnn(inference: DNNInference):
    """Run performance comparison for DNN: Full vs Incremental evaluation"""
    print("\n" + "=" * 70)
    print("DNN Performance Test: Full vs Incremental Evaluation")
    print("=" * 70)

    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # Test 1: Full evaluation
    print("\n1. Full Evaluation - Matrix Multiplication (1000 iterations)")

    start_time = time.time()
    for _ in range(1000):
        features = DNNFeatures.board_to_features(board)
        output = inference.evaluate_full(features)
    full_time = time.time() - start_time

    print(f"   Time: {full_time:.4f} seconds")
    print(f"   Avg per evaluation: {full_time / 1000 * 1000:.3f} ms")

    # Test 2: Incremental evaluation
    print("\n2. Incremental Evaluation - Accumulator Add/Subtract (500 push/pop cycles)")

    updater = DNNIncrementalUpdater(board)
    white_feat, black_feat = updater.get_features_both()
    inference._refresh_accumulator(white_feat, True)
    inference._refresh_accumulator(black_feat, False)

    start_time = time.time()
    for _ in range(500):
        old_white = set(updater.white_features)
        old_black = set(updater.black_features)
        old_perspective = updater.board.turn == chess.WHITE

        updater.push(move)

        new_white = set(updater.white_features)
        new_black = set(updater.black_features)
        new_perspective = updater.board.turn == chess.WHITE

        if new_perspective:
            added = new_white - old_white
            removed = old_white - new_white
        else:
            added = new_black - old_black
            removed = old_black - new_black

        inference.update_accumulator(added, removed, new_perspective)

        current_features = list(new_white) if new_perspective else list(new_black)
        output = inference.evaluate_incremental(current_features, new_perspective)

        updater.pop()

        if old_perspective:
            added_back = old_white - new_white
            removed_back = new_white - old_white
        else:
            added_back = old_black - new_black
            removed_back = new_black - old_black

        inference.update_accumulator(added_back, removed_back, old_perspective)

        old_features = list(old_white) if old_perspective else list(old_black)
        output = inference.evaluate_incremental(old_features, old_perspective)

    incremental_time = time.time() - start_time

    print(f"   Time: {incremental_time:.4f} seconds")
    print(f"   Avg per push/pop cycle: {incremental_time / 500 * 1000:.3f} ms")

    print("\n" + "─" * 70)
    print("Results:")
    print(f"  Full (matrix multiply):     {full_time / 1000 * 1000:.3f} ms per evaluation")
    print(f"  Incremental (add/subtract): {incremental_time / 500 * 1000:.3f} ms per cycle")
    print(f"  Speedup: {full_time / (incremental_time) * 500 / 1000:.2f}x")
    print("\nKey: Incremental uses accumulator (add/subtract weight vectors)")
    print("instead of full matrix multiplication for the first layer.")
    print("=" * 70)


def interactive_loop(inference):
    """Main interactive loop"""
    print("\n" + "=" * 60)
    print("Chess Position Evaluator")
    print("=" * 60)
    print(f"Network type: {nn_type}")
    print(f"Model: {MODEL_PATH}")
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

            result = evaluate_fen(user_input, inference)
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


def main():
    """Main entry point"""
    if nn_type not in ["NNUE", "DNN"]:
        print(f"ERROR: Invalid NN_TYPE '{nn_type}'. Must be 'NNUE' or 'DNN'")
        sys.exit(1)

    # Validate test_type
    # valid_test_types = ["Interactive-FEN", "Incremental-vs-Full", "Accumulator-Correctness", "Eval-Accuracy"]
    if test_type not in valid_test_types:
        print(f"ERROR: Invalid TEST_TYPE '{test_type}'")
        print(f"Must be one of: {', '.join(valid_test_types)}")
        sys.exit(1)

    inference = load_model(MODEL_PATH, nn_type)

    if test_type == "Accumulator-Correctness":
        print(f"\nRunning accumulator correctness test for {nn_type}...")

        if nn_type == "NNUE":
            test_accumulator_correctness_nnue(inference)
        else:
            test_accumulator_correctness_dnn(inference)

    elif test_type == "Eval-Accuracy":
        print(f"\nRunning evaluation accuracy test for {nn_type}...")
        test_eval_accuracy(inference, nn_type)

    elif test_type == "Incremental-vs-Full":
        print(f"\nRunning performance comparison for {nn_type}...")
        print("Comparing: Matrix multiplication vs Accumulator (add/subtract)")

        if nn_type == "NNUE":
            performance_test_nnue(inference)
        else:
            performance_test_dnn(inference)

        print("\nPerformance test complete.")

    else:  # Interactive-FEN
        print("\nTesting with starting position...")
        startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = evaluate_fen(startpos, inference)
        print_evaluation(result)

        interactive_loop(inference)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[1] == "DNN" or sys.argv[1] == "NNUE":
            nn_type = sys.argv[1]
        if sys.argv[2].isdigit() and 0 <= int(sys.argv[2]) < len(valid_test_types_dict):
            test_type = valid_test_types_dict[int(sys.argv[2])]

    if nn_type is None or test_type is None:
        print(f"Usage: {os.path.basename(sys.argv[0])} NN-TYPE {'{DNN, NNUE}'}, Test-Types {valid_test_types_dict}")
        print(f"Example: {os.path.basename(sys.argv[0])} DNN 0")
        exit()

    main()
