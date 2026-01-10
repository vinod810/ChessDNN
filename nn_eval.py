"""
Chess Neural Network Evaluation Script
Loads trained NNUE or DNN models and evaluates positions from FEN strings.
Uses numpy for fast inference (avoiding PyTorch overhead).
"""

import numpy as np
import chess
import torch
import sys
from typing import List

# Import configuration and classes from train_nn
# Note: Adjust the import path if train_nn.py is in a different location
try:
    from train_nn import (
        NN_TYPE, MODEL_PATH, TANH_SCALE,
        NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE,
        DNN_INPUT_SIZE, DNN_HIDDEN_LAYERS,
        NNUEFeatures, DNNFeatures,
        NNUENetwork, DNNNetwork
    )
except ImportError:
    print("ERROR: Could not import from train_nn.py")
    print("Make sure train_nn.py (or train_nn_fixed.py) is in the same directory")
    sys.exit(1)


class NNUEInference:
    """Numpy-based inference engine for NNUE"""

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

    @staticmethod
    def clipped_relu(x):
        """Clipped ReLU activation [0, 1]"""
        return np.clip(x, 0, 1)

    def evaluate(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        """
        Evaluate position from side-to-move's perspective.

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

        # Set active features to 1.0
        for f in white_features:
            if 0 <= f < len(white_input):
                white_input[f] = 1.0
        for f in black_features:
            if 0 <= f < len(black_input):
                black_input[f] = 1.0

        # Feature transform (shared weights)
        white_hidden = self.clipped_relu(np.dot(white_input, self.ft_weight.T) + self.ft_bias)
        black_hidden = self.clipped_relu(np.dot(black_input, self.ft_weight.T) + self.ft_bias)

        # Perspective concatenation
        if stm:  # White to move
            hidden = np.concatenate([white_hidden, black_hidden])
        else:  # Black to move
            hidden = np.concatenate([black_hidden, white_hidden])

        # Further hidden layers
        x = self.clipped_relu(np.dot(hidden, self.l1_weight.T) + self.l1_bias)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        output = np.dot(x, self.l3_weight.T) + self.l3_bias

        return output[0]  # Linear output

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board position"""
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate(white_feat, black_feat, board.turn == chess.WHITE)


class DNNInference:
    """Numpy-based inference engine for DNN"""

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

    @staticmethod
    def clipped_relu(x):
        """Clipped ReLU activation [0, 1]"""
        return np.clip(x, 0, 1)

    def evaluate(self, features: List[int]) -> float:
        """
        Evaluate position from side-to-move's perspective.

        Args:
            features: Active feature indices (768-dimensional)

        Returns:
            Evaluation score (linear output, approximately in [-1, 1])
        """
        # Create sparse input vector
        feature_input = np.zeros(DNN_INPUT_SIZE, dtype=np.float32)

        # Set active features to 1.0
        for f in features:
            if 0 <= f < len(feature_input):
                feature_input[f] = 1.0

        # Forward pass through hidden layers
        x = self.clipped_relu(np.dot(feature_input, self.l1_weight.T) + self.l1_bias)
        x = self.clipped_relu(np.dot(x, self.l2_weight.T) + self.l2_bias)
        x = self.clipped_relu(np.dot(x, self.l3_weight.T) + self.l3_bias)
        output = np.dot(x, self.l4_weight.T) + self.l4_bias

        return output[0]  # Linear output

    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate a chess board position"""
        feat = DNNFeatures.board_to_features(board)
        return self.evaluate(feat)


def load_model(model_path: str, nn_type: str):
    """
    Load trained model from checkpoint file.

    Args:
        model_path: Path to .pt checkpoint file
        nn_type: "NNUE" or "DNN"

    Returns:
        Inference engine (NNUEInference or DNNInference)
    """
    print(f"Loading {nn_type} model from {model_path}...")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Create model
        if nn_type == "NNUE":
            model = NNUENetwork()
        elif nn_type == "DNN":
            model = DNNNetwork()
        else:
            raise ValueError(f"Unknown NN_TYPE: {nn_type}. Must be 'NNUE' or 'DNN'")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Checkpoint contains metadata (epoch, model_state_dict, val_loss, etc.)
                state_dict = checkpoint['model_state_dict']
                print(f"  Checkpoint info: Epoch {checkpoint.get('epoch', 'unknown')}, "
                      f"Val loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
            elif 'state_dict' in checkpoint:
                # Alternative format
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            # Not a dict, might be an old format
            state_dict = checkpoint

        # Load weights
        model.load_state_dict(state_dict)
        model.eval()

        # Create numpy inference engine
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
    """
    Convert linear network output to centipawns.

    Args:
        output: Network output (approximately in [-1, 1])

    Returns:
        Evaluation in centipawns
    """
    # Clamp to avoid arctanh domain errors
    output = np.clip(output, -0.99, 0.99)

    # Convert: cp = arctanh(output) * TANH_SCALE
    centipawns = np.arctanh(output) * TANH_SCALE

    return centipawns


def evaluate_fen(fen: str, inference) -> dict:
    """
    Evaluate a position from FEN string.

    Args:
        fen: FEN string
        inference: Inference engine (NNUEInference or DNNInference)

    Returns:
        Dictionary with evaluation results
    """
    try:
        # Parse FEN
        board = chess.Board(fen)

        # Evaluate
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
        return {
            'success': False,
            'error': f"Invalid FEN: {e}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Evaluation error: {e}"
        }


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

    # Convert to pawns for readability
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

    # Pre-defined FEN strings
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    while True:
        try:
            # Get input
            user_input = input("FEN> ").strip()

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            # Check for help
            if user_input.lower() == 'help':
                print_help()
                continue

            # Check for empty input
            if not user_input:
                continue

            # Handle 'startpos' shortcut
            if user_input.lower() == 'startpos':
                user_input = startpos_fen
                print(f"Using starting position: {startpos_fen}")

            # Evaluate position
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
    # Validate configuration
    if NN_TYPE not in ["NNUE", "DNN"]:
        print(f"ERROR: Invalid NN_TYPE '{NN_TYPE}'. Must be 'NNUE' or 'DNN'")
        print("Please check the NN_TYPE setting in train_nn.py")
        sys.exit(1)

    # Load model
    inference = load_model(MODEL_PATH, NN_TYPE)

    # Test with starting position
    print("\nTesting with starting position...")
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    result = evaluate_fen(startpos, inference)
    print_evaluation(result)

    # Enter interactive loop
    interactive_loop(inference)


if __name__ == "__main__":
    main()