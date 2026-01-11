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

import numpy as np
import chess
import torch
import sys
import time
from typing import List, Set, Tuple

# Import configuration and classes from train_nn
try:
    from train_nn import (
        NN_TYPE, MODEL_PATH, TANH_SCALE,
        NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE,
        DNN_INPUT_SIZE, DNN_HIDDEN_LAYERS,
        NNUEFeatures, DNNFeatures,
        NNUENetwork, DNNNetwork,
        IncrementalFeatureUpdater
    )
except ImportError:
    print("ERROR: Could not import from train_nn.py")
    print("Make sure train_nn.py (or train_nn_fixed.py) is in the same directory")
    sys.exit(1)

# Configuration
# Allowed values: "Interactive-FEN", "Incremental-vs-Full", "Accumulator-Correctness", "Eval-accuracy"
TEST_TYPE = "Eval-accuracy"


class DNNIncrementalUpdater:
    """
    Incrementally maintains DNN features for both perspectives.
    Similar to NNUE's IncrementalFeatureUpdater, but for DNN's 768-dimensional features.
    """

    def __init__(self, board: chess.Board):
        """Initialize with a board position"""
        self.board = board.copy()
        self.white_features: Set[int] = set(DNNFeatures.extract_features(board, chess.WHITE))
        self.black_features: Set[int] = set(DNNFeatures.extract_features(board, chess.BLACK))

    def _get_feature_for_perspective(self, perspective: bool, square: int,
                                     piece_type: int, piece_color: bool) -> int:
        """Get feature index for a piece from a given perspective"""
        if not perspective:
            rank = square // 8
            file = square % 8
            square = (7 - rank) * 8 + file

        piece_is_friendly = (piece_color == chess.WHITE) == perspective
        piece_idx = DNNFeatures.get_piece_index(piece_type, piece_is_friendly)
        return square * 12 + piece_idx

    def _remove_piece_features(self, square: int, piece_type: int, piece_color: bool):
        """Remove features for a piece at the given square"""
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        self.white_features.discard(white_feat)
        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        self.black_features.discard(black_feat)

    def _add_piece_features(self, square: int, piece_type: int, piece_color: bool):
        """Add features for a piece at the given square"""
        white_feat = self._get_feature_for_perspective(True, square, piece_type, piece_color)
        self.white_features.add(white_feat)
        black_feat = self._get_feature_for_perspective(False, square, piece_type, piece_color)
        self.black_features.add(black_feat)

    def push(self, move: chess.Move):
        """Update features after making a move"""
        from_sq = move.from_square
        to_sq = move.to_square

        piece = self.board.piece_at(from_sq)
        if piece is None:
            self.board.push(move)
            return

        moving_piece_type = piece.piece_type
        moving_piece_color = piece.color

        captured_piece = self.board.piece_at(to_sq)
        is_en_passant = self.board.is_en_passant(move)

        if is_en_passant:
            ep_sq = to_sq + (-8 if moving_piece_color == chess.WHITE else 8)
            captured_piece = self.board.piece_at(ep_sq)
            if captured_piece:
                self._remove_piece_features(ep_sq, captured_piece.piece_type, captured_piece.color)

        if captured_piece and not is_en_passant:
            self._remove_piece_features(to_sq, captured_piece.piece_type, captured_piece.color)

        is_castling = self.board.is_castling(move)
        if is_castling:
            if to_sq > from_sq:
                rook_from = chess.H1 if moving_piece_color == chess.WHITE else chess.H8
                rook_to = chess.F1 if moving_piece_color == chess.WHITE else chess.F8
            else:
                rook_from = chess.A1 if moving_piece_color == chess.WHITE else chess.A8
                rook_to = chess.D1 if moving_piece_color == chess.WHITE else chess.D8
            self._remove_piece_features(rook_from, chess.ROOK, moving_piece_color)

        self._remove_piece_features(from_sq, moving_piece_type, moving_piece_color)
        self.board.push(move)

        if move.promotion:
            moving_piece_type = move.promotion

        self._add_piece_features(to_sq, moving_piece_type, moving_piece_color)

        if is_castling:
            self._add_piece_features(rook_to, chess.ROOK, moving_piece_color)

    def pop(self):
        """Undo the last move"""
        self.board.pop()
        self.white_features = set(DNNFeatures.extract_features(self.board, chess.WHITE))
        self.black_features = set(DNNFeatures.extract_features(self.board, chess.BLACK))

    def get_features(self) -> List[int]:
        """Get features for current side to move"""
        if self.board.turn == chess.WHITE:
            return list(self.white_features)
        else:
            return list(self.black_features)

    def get_features_both(self) -> Tuple[List[int], List[int]]:
        """Get features for both perspectives"""
        return list(self.white_features), list(self.black_features)


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

    # Test game: 1. e4 d5 2. exd5 Bh3 3. Bb5+ Nd7 4. Nf3 e5 5. dxe6 Qg5
    # 6. O-O O-O-O 7. e7 Ngf6 8. e8=Q Bb4 9. Nd4 Rhxe8 10. Nb3 Bxd2 11. N1xd2
    moves_san = [
        "e4", "d5", "exd5", "Bh3", "Bb5+", "Nd7", "Nf3", "e5",
        "dxe6", "Qg5", "O-O", "O-O-O", "e7", "Ngf6", "e8=Q", "Bb4",
        "Nd4", "Rhxe8", "Nb3", "Bxd2", "N1xd2"
    ]

    print(f"\nPlaying game: 1. e4 d5 2. exd5 Bh3 3. Bb5+ Nd7 4. Nf3 e5")
    print(f"              5. dxe6 Qg5 6. O-O O-O-O 7. e7 Ngf6 8. e8=Q Bb4")
    print(f"              9. Nd4 Rhxe8 10. Nb3 Bxd2 11. N1xd2")

    # Play the game
    board = chess.Board()
    updater = IncrementalFeatureUpdater(board)

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

    print(f"\n✓ Played {move_count} moves")

    # Test 1: After move 11
    print("\n" + "─" * 70)
    print("Test 1: After move 11 (N1xd2)")
    print("─" * 70)

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
    for i in range(8):
        updater.board.pop()

    # Refresh accumulator after pops
    white_feat, black_feat = NNUEFeatures.board_to_features(updater.board)
    inference._refresh_accumulator(white_feat, black_feat)
    stm = updater.board.turn == chess.WHITE

    # Evaluate with incremental
    eval_incremental = inference.evaluate_incremental(white_feat, black_feat, stm)

    # Evaluate with full
    eval_full = inference.evaluate_full(white_feat, black_feat, stm)

    # Compare
    diff = abs(eval_incremental - eval_full)

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

    moves_san = [
        "e4", "d5", "exd5", "Bh3", "Bb5+", "Nd7", "Nf3", "e5",
        "dxe6", "Qg5", "O-O", "O-O-O", "e7", "Ngf6", "e8=Q", "Bb4",
        "Nd4", "Rhxe8", "Nb3", "Bxd2", "N1xd2"
    ]

    print(f"\nPlaying game: 1. e4 d5 2. exd5 Bh3 3. Bb5+ Nd7 4. Nf3 e5")
    print(f"              5. dxe6 Qg5 6. O-O O-O-O 7. e7 Ngf6 8. e8=Q Bb4")
    print(f"              9. Nd4 Rhxe8 10. Nb3 Bxd2 11. N1xd2")

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

        # Track old features
        old_perspective = updater.board.turn == chess.WHITE
        if old_perspective:
            old_features = set(updater.white_features)
        else:
            old_features = set(updater.black_features)

        # Push move
        updater.push(move)

        # Get new features for new perspective
        new_perspective = updater.board.turn == chess.WHITE
        if new_perspective:
            new_features = set(updater.white_features)
        else:
            new_features = set(updater.black_features)

        # Update accumulator for the new perspective
        inference.update_accumulator(
            new_features - old_features,
            old_features - new_features,
            new_perspective
        )

        move_count += 1

    print(f"\n✓ Played {move_count} moves")

    # Test 1: After move 11
    print("\n" + "─" * 70)
    print("Test 1: After move 11 (N1xd2)")
    print("─" * 70)

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
    for i in range(8):
        updater.pop()

    # Refresh accumulators after pops
    white_feat, black_feat = updater.get_features_both()
    inference._refresh_accumulator(white_feat, True)
    inference._refresh_accumulator(black_feat, False)

    features = updater.get_features()
    perspective = updater.board.turn == chess.WHITE

    # Evaluate with incremental
    eval_incremental = inference.evaluate_incremental(features, perspective)

    # Evaluate with full
    features_full = DNNFeatures.board_to_features(updater.board)
    eval_full = inference.evaluate_full(features_full)

    # Compare
    diff = abs(eval_incremental - eval_full)

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
    Test evaluation accuracy against ground truth from CSV file.
    Computes MSE of tanh(cp/400) between predicted and true scores.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Evaluation Accuracy Test")
    print("=" * 70)

    csv_path = "pgn/chessData.csv" #random_fen.csv"

    # Try to read CSV file
    import csv
    import os

    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: CSV file not found: {csv_path}")
        print("Please ensure the file exists")
        print("\nExpected format (comma-separated):")
        print("  FEN,Evaluation")
        print("  rnbqkb1r/pppppppp/B4n2/8/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2,-459")
        return

    print(f"\nReading positions from: {csv_path}")

    positions = []
    try:
        with open(csv_path, 'r') as f:
            # Standard CSV with comma delimiter
            reader = csv.DictReader(f)

            # Debug: Show available columns
            available_columns = reader.fieldnames
            print(f"CSV columns found: {available_columns}")

            for i, row in enumerate(reader):
                if i >= 1000:  # Only first 1000 positions
                    break

                # Try to find FEN and Evaluation columns
                fen = None
                score = None

                # Try exact matches first (FEN, Evaluation)
                if 'FEN' in row:
                    fen = row['FEN'].strip()
                elif 'fen' in row:
                    fen = row['fen'].strip()

                if 'Evaluation' in row:
                    try:
                        score = float(row['Evaluation'].strip())
                        #print(score)
                    except (ValueError, AttributeError):
                        pass
                elif 'evaluation' in row:
                    try:
                        score = float(row['evaluation'].strip())
                    except (ValueError, AttributeError):
                        pass
                elif 'score' in row:
                    try:
                        score = float(row['score'].strip())
                    except (ValueError, AttributeError):
                        pass
                elif 'Score' in row:
                    try:
                        score = float(row['Score'].strip())
                    except (ValueError, AttributeError):
                        pass

                # If exact match failed, try partial matches
                if fen is None or score is None:
                    for key in row.keys():
                        key_lower = key.lower().strip()
                        if fen is None and 'fen' in key_lower:
                            fen = row[key].strip()
                        if score is None and ('eval' in key_lower or 'score' in key_lower or 'cp' in key_lower):
                            try:
                                score = float(row[key].strip())
                            except (ValueError, AttributeError):
                                continue

                if fen and score is not None:
                    board = chess.Board(fen)
                    score = score if board.turn == chess.WHITE else -score
                    #print(score)
                    positions.append({'fen': fen, 'true_cp': score})
                elif i < 3:  # Show first few problematic rows for debugging
                    print(f"  Warning: Row {i+1} - fen={'found' if fen else 'missing'}, score={'found' if score is not None else 'missing'}")

        if not positions:
            print("\n❌ ERROR: No valid positions found in CSV")
            print(f"Available columns: {available_columns}")
            print("\nPlease ensure CSV has:")
            print("  - A column named 'FEN' (for FEN strings)")
            print("  - A column named 'Evaluation' (for centipawn scores)")
            return

        print(f"✓ Loaded {len(positions)} positions")

    except Exception as e:
        print(f"❌ ERROR reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate all positions
    print("\nEvaluating positions...")

    true_tanh_values = []
    pred_tanh_values = []
    errors = []

    for i, pos in enumerate(positions):
        if (i + 1) % 100 == 0:
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
        print(f"{i+1:2d}. True: {true_cp:+7.1f} cp | Pred: {pred_cp:+7.1f} cp | Diff: {diff:+7.1f} cp")

    print("\n" + "=" * 70)
    print("Evaluation accuracy test complete!")
    print("=" * 70)

    # Evaluate all positions
    print("\nEvaluating positions...")

    true_tanh_values = []
    pred_tanh_values = []
    errors = []

    # for i, pos in enumerate(positions):
    #     if (i + 1) % 100 == 0:
    #         print(f"  Progress: {i + 1}/{len(positions)}")
    #
    #     try:
    #         # Get true tanh value
    #         true_cp = pos['true_cp']
    #         true_tanh = np.tanh(true_cp / TANH_SCALE)
    #
    #         # Get predicted value
    #         board = chess.Board(pos['fen'])
    #
    #         # Use full evaluation for accuracy test
    #         if nn_type == "NNUE":
    #             white_feat, black_feat = NNUEFeatures.board_to_features(board)
    #             pred_output = inference.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)
    #         else:  # DNN
    #             features = DNNFeatures.board_to_features(board)
    #             pred_output = inference.evaluate_full(features)
    #
    #         # pred_output is already in tanh space (network outputs ~tanh(cp/400))
    #         pred_tanh = pred_output
    #
    #         true_tanh_values.append(true_tanh)
    #         pred_tanh_values.append(pred_tanh)
    #
    #     except Exception as e:
    #         errors.append((i, pos['fen'], str(e)))
    #
    # if errors:
    #     print(f"\n⚠ {len(errors)} positions failed to evaluate:")
    #     for idx, fen, error in errors[:5]:  # Show first 5 errors
    #         print(f"  Position {idx}: {error[:60]}")
    #     if len(errors) > 5:
    #         print(f"  ... and {len(errors) - 5} more")
    #
    # # Compute MSE
    # true_tanh_values = np.array(true_tanh_values)
    # pred_tanh_values = np.array(pred_tanh_values)
    #
    # mse = np.mean((true_tanh_values - pred_tanh_values) ** 2)
    # rmse = np.sqrt(mse)
    # mae = np.mean(np.abs(true_tanh_values - pred_tanh_values))
    #
    # # Also compute MSE in centipawn space for reference
    # true_cp_values = true_tanh_values * TANH_SCALE
    # pred_cp_values = np.arctanh(np.clip(pred_tanh_values, -0.99, 0.99)) * TANH_SCALE
    # mse_cp = np.mean((true_cp_values - pred_cp_values) ** 2)
    # rmse_cp = np.sqrt(mse_cp)
    # mae_cp = np.mean(np.abs(true_cp_values - pred_cp_values))
    #
    # # Results
    # print("\n" + "─" * 70)
    # print("Results:")
    # print("─" * 70)
    # print(f"Positions evaluated: {len(pred_tanh_values)}")
    # print(f"Positions failed:    {len(errors)}")
    # print()
    # print("Tanh Space (network output space):")
    # print(f"  MSE:  {mse:.6f}")
    # print(f"  RMSE: {rmse:.6f}")
    # print(f"  MAE:  {mae:.6f}")
    # print()
    # print("Centipawn Space (for reference):")
    # print(f"  MSE:  {mse_cp:.2f}")
    # print(f"  RMSE: {rmse_cp:.2f} cp")
    # print(f"  MAE:  {mae_cp:.2f} cp")
    #
    # # Show some example predictions
    # print("\n" + "─" * 70)
    # print("Sample predictions (first 10):")
    # print("─" * 70)
    # for i in range(min(10, len(positions))):
    #     true_cp = positions[i]['true_cp']
    #     pred_cp = pred_cp_values[i]
    #     diff = pred_cp - true_cp
    #     print(f"{i+1:2d}. True: {true_cp:+7.1f} cp | Pred: {pred_cp:+7.1f} cp | Diff: {diff:+7.1f} cp")
    #
    # print("\n" + "=" * 70)
    # print("Evaluation accuracy test complete!")
    # print("=" * 70)
    #
    # # Evaluate all positions
    # print("\nEvaluating positions...")
    #
    # true_tanh_values = []
    # pred_tanh_values = []
    # errors = []
    #
    # for i, pos in enumerate(positions):
    #     if (i + 1) % 100 == 0:
    #         print(f"  Progress: {i + 1}/{len(positions)}")
    #
    #     try:
    #         # Get true tanh value
    #         true_cp = pos['true_cp']
    #         true_tanh = np.tanh(true_cp / TANH_SCALE)
    #
    #         # Get predicted value
    #         board = chess.Board(pos['fen'])
    #
    #         # Use full evaluation for accuracy test
    #         if nn_type == "NNUE":
    #             white_feat, black_feat = NNUEFeatures.board_to_features(board)
    #             pred_output = inference.evaluate_full(white_feat, black_feat, board.turn == chess.WHITE)
    #         else:  # DNN
    #             features = DNNFeatures.board_to_features(board)
    #             pred_output = inference.evaluate_full(features)
    #
    #         # pred_output is already in tanh space (network outputs ~tanh(cp/400))
    #         pred_tanh = pred_output
    #
    #         true_tanh_values.append(true_tanh)
    #         pred_tanh_values.append(pred_tanh)
    #
    #     except Exception as e:
    #         errors.append((i, pos['fen'], str(e)))
    #
    # if errors:
    #     print(f"\n⚠ {len(errors)} positions failed to evaluate:")
    #     for idx, fen, error in errors[:5]:  # Show first 5 errors
    #         print(f"  Position {idx}: {error}")
    #     if len(errors) > 5:
    #         print(f"  ... and {len(errors) - 5} more")
    #
    # # Compute MSE
    # true_tanh_values = np.array(true_tanh_values)
    # pred_tanh_values = np.array(pred_tanh_values)
    #
    # mse = np.mean((true_tanh_values - pred_tanh_values) ** 2)
    # rmse = np.sqrt(mse)
    # mae = np.mean(np.abs(true_tanh_values - pred_tanh_values))
    #
    # # Also compute MSE in centipawn space for reference
    # true_cp_values = true_tanh_values * TANH_SCALE
    # pred_cp_values = np.arctanh(np.clip(pred_tanh_values, -0.99, 0.99)) * TANH_SCALE
    # mse_cp = np.mean((true_cp_values - pred_cp_values) ** 2)
    # rmse_cp = np.sqrt(mse_cp)
    # mae_cp = np.mean(np.abs(true_cp_values - pred_cp_values))
    #
    # # Results
    # print("\n" + "─" * 70)
    # print("Results:")
    # print("─" * 70)
    # print(f"Positions evaluated: {len(pred_tanh_values)}")
    # print(f"Positions failed:    {len(errors)}")
    # print()
    # print("Tanh Space (network output space):")
    # print(f"  MSE:  {mse:.6f}")
    # print(f"  RMSE: {rmse:.6f}")
    # print(f"  MAE:  {mae:.6f}")
    # print()
    # print("Centipawn Space (for reference):")
    # print(f"  MSE:  {mse_cp:.2f}")
    # print(f"  RMSE: {rmse_cp:.2f} cp")
    # print(f"  MAE:  {mae_cp:.2f} cp")
    #
    # # Show some example predictions
    # print("\n" + "─" * 70)
    # print("Sample predictions (first 5):")
    # print("─" * 70)
    # for i in range(min(5, len(positions))):
    #     true_cp = positions[i]['true_cp']
    #     pred_cp = pred_cp_values[i]
    #     diff = pred_cp - true_cp
    #     print(f"{i+1}. True: {true_cp:+7.1f} cp | Pred: {pred_cp:+7.1f} cp | Diff: {diff:+7.1f} cp")
    #
    # print("\n" + "=" * 70)
    # print("Evaluation accuracy test complete!")
    # print("=" * 70)


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
    print(f"   Avg per evaluation: {full_time/1000*1000:.3f} ms")

    # Test 2: Incremental evaluation
    print("\n2. Incremental Evaluation - Accumulator Add/Subtract (500 push/pop cycles)")

    updater = IncrementalFeatureUpdater(board)
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
    print(f"   Avg per push/pop cycle: {incremental_time/500*1000:.3f} ms")

    print("\n" + "─" * 70)
    print("Results:")
    print(f"  Full (matrix multiply):     {full_time/1000*1000:.3f} ms per evaluation")
    print(f"  Incremental (add/subtract): {incremental_time/500*1000:.3f} ms per cycle")
    print(f"  Speedup: {full_time/(incremental_time)*500/1000:.2f}x")
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
    print(f"   Avg per evaluation: {full_time/1000*1000:.3f} ms")

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
    print(f"   Avg per push/pop cycle: {incremental_time/500*1000:.3f} ms")

    print("\n" + "─" * 70)
    print("Results:")
    print(f"  Full (matrix multiply):     {full_time/1000*1000:.3f} ms per evaluation")
    print(f"  Incremental (add/subtract): {incremental_time/500*1000:.3f} ms per cycle")
    print(f"  Speedup: {full_time/(incremental_time)*500/1000:.2f}x")
    print("\nKey: Incremental uses accumulator (add/subtract weight vectors)")
    print("instead of full matrix multiplication for the first layer.")
    print("=" * 70)


def interactive_loop(inference):
    """Main interactive loop"""
    print("\n" + "=" * 60)
    print("Chess Position Evaluator")
    print("=" * 60)
    print(f"Network type: {NN_TYPE}")
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
    if NN_TYPE not in ["NNUE", "DNN"]:
        print(f"ERROR: Invalid NN_TYPE '{NN_TYPE}'. Must be 'NNUE' or 'DNN'")
        sys.exit(1)

    # Validate TEST_TYPE
    valid_test_types = ["Interactive-FEN", "Incremental-vs-Full", "Accumulator-Correctness", "Eval-accuracy"]
    if TEST_TYPE not in valid_test_types:
        print(f"ERROR: Invalid TEST_TYPE '{TEST_TYPE}'")
        print(f"Must be one of: {', '.join(valid_test_types)}")
        sys.exit(1)

    inference = load_model(MODEL_PATH, NN_TYPE)

    if TEST_TYPE == "Accumulator-Correctness":
        print(f"\nRunning accumulator correctness test for {NN_TYPE}...")

        if NN_TYPE == "NNUE":
            test_accumulator_correctness_nnue(inference)
        else:
            test_accumulator_correctness_dnn(inference)

    elif TEST_TYPE == "Eval-accuracy":
        print(f"\nRunning evaluation accuracy test for {NN_TYPE}...")
        test_eval_accuracy(inference, NN_TYPE)

    elif TEST_TYPE == "Incremental-vs-Full":
        print(f"\nRunning performance comparison for {NN_TYPE}...")
        print("Comparing: Matrix multiplication vs Accumulator (add/subtract)")

        if NN_TYPE == "NNUE":
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
    main()