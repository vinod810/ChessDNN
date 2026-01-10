import numpy as np
import chess
import torch
import sys
import os

# Import from the main training file
try:
    from halfkp import NNUENetwork, NNUEFeatures, INPUT_SIZE, HIDDEN_SIZE # FIXME
except ImportError:
    try:
        #from nnue_train_parallel import NNUENetwork, NNUEFeatures, INPUT_SIZE, FIRST_HIDDEN_SIZE
        pass
    except ImportError:
        print("Error: Could not import from halfkp.py or nnue_train_parallel.py")
        print("Make sure one of these files is in the same directory")
        sys.exit(1)


class NNUEInference:
    """Numpy-based inference engine that can load from .pt checkpoint"""

    def __init__(self, model: NNUENetwork):
        """Initialize from PyTorch model"""
        model.eval()
        self.input_size = model.input_size
        self.hidden_size = model.hidden_size
        self.ft_weight = model.ft.weight.detach().cpu().numpy()
        self.ft_bias = model.ft.bias.detach().cpu().numpy()
        self.l1_weight = model.l1.weight.detach().cpu().numpy()
        self.l1_bias = model.l1.bias.detach().cpu().numpy()
        self.l2_weight = model.l2.weight.detach().cpu().numpy()
        self.l2_bias = model.l2.bias.detach().cpu().numpy()
        self.l3_weight = model.l3.weight.detach().cpu().numpy()
        self.l3_bias = model.l3.bias.detach().cpu().numpy()

    def clipped_relu(self, x):
        """ClippedReLU activation: clip to [0, 1]"""
        return np.clip(x, 0, 1)

    def evaluate(self, white_features: list, black_features: list, stm: bool) -> float:
        """Evaluate position using numpy"""
        white_input = np.zeros(self.input_size)
        black_input = np.zeros(self.input_size)

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
        """Evaluate a chess board position"""
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate(white_feat, black_feat, board.turn == chess.WHITE)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, print_summary: bool = True):
        """
        Load model from PyTorch .pt checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint file
            print_summary: Whether to print model summary

        Returns:
            NNUEInference instance
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Create model and load state dict
        model = NNUENetwork(INPUT_SIZE, HIDDEN_SIZE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if print_summary:
            cls._print_checkpoint_summary(checkpoint_path, checkpoint, model)

        return cls(model)

    @classmethod
    def load_from_bin(cls, bin_path: str, print_summary: bool = True):
        """
        Load model from binary .bin weights file (legacy format).

        Args:
            bin_path: Path to the .bin weights file
            print_summary: Whether to print model summary

        Returns:
            NNUEInference instance
        """
        import struct

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Weights file not found: {bin_path}")

        with open(bin_path, 'rb') as f:
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

            inference.input_size = input_size
            inference.hidden_size = hidden_size

        if print_summary:
            cls._print_bin_summary(bin_path, inference)

        return inference

    @staticmethod
    def _print_checkpoint_summary(path: str, checkpoint: dict, model: NNUENetwork):
        """Print summary of loaded .pt checkpoint"""
        file_size = os.path.getsize(path)

        print("\n" + "=" * 70)
        print("MODEL CHECKPOINT SUMMARY")
        print("=" * 70)
        print(f"File: {path}")
        print(f"File size: {file_size / 1024 / 1024:.2f} MB")
        print("-" * 70)

        # Checkpoint metadata
        print("\nCheckpoint Metadata:")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation Loss: {checkpoint['val_loss']:.6f}")

        # Model architecture
        print("\nModel Architecture:")
        print(f"  Input size: {model.input_size:,}")
        print(f"  Hidden size: {model.hidden_size}")
        print(f"  Feature transformer: {model.input_size:,} -> {model.hidden_size}")
        print(f"  Layer 1: {model.hidden_size * 2} -> 32")
        print(f"  Layer 2: 32 -> 32")
        print(f"  Output: 32 -> 1")

        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\nParameter Counts:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Layer-wise breakdown
        print("\nLayer-wise Parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel():,} params)")

        # Weight statistics
        print("\nWeight Statistics:")
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy()
            print(f"  {name}:")
            print(f"    min={data.min():.4f}, max={data.max():.4f}, "
                  f"mean={data.mean():.4f}, std={data.std():.4f}")

        print("=" * 70 + "\n")

    @staticmethod
    def _print_bin_summary(path: str, inference: 'NNUEInference'):
        """Print summary of loaded .bin weights file"""
        file_size = os.path.getsize(path)

        print("\n" + "=" * 70)
        print("MODEL WEIGHTS SUMMARY (.bin format)")
        print("=" * 70)
        print(f"File: {path}")
        print(f"File size: {file_size / 1024 / 1024:.2f} MB")
        print("-" * 70)

        print("\nModel Architecture:")
        print(f"  Input size: {inference.input_size:,}")
        print(f"  Hidden size: {inference.hidden_size}")

        # Calculate total parameters
        total_params = (
                inference.ft_weight.size + inference.ft_bias.size +
                inference.l1_weight.size + inference.l1_bias.size +
                inference.l2_weight.size + inference.l2_bias.size +
                inference.l3_weight.size + inference.l3_bias.size
        )
        print(f"\nTotal parameters: {total_params:,}")

        print("\nWeight Statistics:")
        weights = [
            ('ft.weight', inference.ft_weight),
            ('ft.bias', inference.ft_bias),
            ('l1.weight', inference.l1_weight),
            ('l1.bias', inference.l1_bias),
            ('l2.weight', inference.l2_weight),
            ('l2.bias', inference.l2_bias),
            ('l3.weight', inference.l3_weight),
            ('l3.bias', inference.l3_bias),
        ]
        for name, data in weights:
            print(f"  {name}: shape={data.shape}")
            print(f"    min={data.min():.4f}, max={data.max():.4f}, "
                  f"mean={data.mean():.4f}, std={data.std():.4f}")

        print("=" * 70 + "\n")


def load_inference(weights_path: str = None) -> NNUEInference:
    """
    Load inference engine, auto-detecting file format.

    Args:
        weights_path: Path to weights file. If None, searches for default files.

    Returns:
        NNUEInference instance
    """
    # Default file search order
    default_files = [
        "best_model.pt",
        "nnue_weights.pt",
        "nnue_weights.bin",
        "model.pt"
    ]

    if weights_path is None:
        # Search for default files
        for filename in default_files:
            if os.path.exists(filename):
                weights_path = filename
                break

        if weights_path is None:
            raise FileNotFoundError(
                f"No weights file found. Searched for: {', '.join(default_files)}\n"
                "Please train the model first or specify a weights file path."
            )

    # Detect format and load
    if weights_path.endswith('.pt'):
        return NNUEInference.load_from_checkpoint(weights_path)
    elif weights_path.endswith('.bin'):
        return NNUEInference.load_from_bin(weights_path)
    else:
        # Try to detect format
        try:
            return NNUEInference.load_from_checkpoint(weights_path)
        except:
            return NNUEInference.load_from_bin(weights_path)


def format_evaluation(eval_score: float) -> str:
    """Format evaluation score for display"""
    clipped_score = np.clip(eval_score, -0.9999, 0.9999)
    cp = np.arctanh(clipped_score) * 400

    if abs(cp) < 10:
        return f"{eval_score:+.4f} (tanh) = {cp:+.1f} cp"
    else:
        return f"{eval_score:+.4f} (tanh) = {cp:+.0f} cp"


def display_board(board: chess.Board):
    """Display the chess board"""
    print("\n" + str(board))
    print(f"\nFEN: {board.fen()}")
    print(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")


def test_nnue_interactive(weights_path: str = None):
    """Interactive testing interface for NNUE evaluation"""

    print("Loading NNUE model...")
    try:
        inference = load_inference(weights_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("=" * 70)
    print("NNUE Chess Position Evaluator")
    print("=" * 70)
    print("\nEnter FEN positions to evaluate.")
    print("Commands:")
    print("  - Enter a FEN string to evaluate")
    print("  - 'start' or 'startpos' for starting position")
    print("  - 'exit' or 'quit' to quit")
    print("=" * 70)

    # Test with starting position first
    print("\n[Example] Evaluating starting position...")
    board = chess.Board()
    display_board(board)
    eval_score = inference.evaluate_board(board)
    print(f"\nEvaluation: {format_evaluation(eval_score)}")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter FEN (or 'exit'/'quit'): ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nExiting. Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ['start', 'startpos', 'starting']:
                board = chess.Board()
            else:
                try:
                    board = chess.Board(user_input)
                except ValueError as e:
                    print(f"\nError: Invalid FEN string!")
                    print(f"Details: {e}")
                    print("\nExample valid FEN:")
                    print("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                    continue

            display_board(board)

            if board.is_valid():
                white_feat, black_feat = NNUEFeatures.board_to_features(board)
                eval_score = inference.evaluate_board(board)

                print(f"\nEvaluation: {format_evaluation(eval_score)}")

                print(f"\nFeature counts:")
                print(f"  White perspective: {len(white_feat)} active features")
                print(f"  Black perspective: {len(black_feat)} active features")

                if eval_score > 0.3:
                    print(f"Position favors {'White' if board.turn else 'Black'}")
                elif eval_score < -0.3:
                    print(f"Position favors {'Black' if board.turn else 'White'}")
                else:
                    print("Position is roughly equal")
            else:
                print("\nWarning: Position may be invalid (illegal king placement, etc.)")
                print("Evaluating anyway...")
                eval_score = inference.evaluate_board(board)
                print(f"\nEvaluation: {format_evaluation(eval_score)}")

            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Please try again or type 'exit' to quit.")


def test_predefined_positions(weights_path: str = None):
    """Test with some predefined interesting positions"""

    print("Loading NNUE model...")
    try:
        inference = load_inference(weights_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_positions = [
        ("Starting position", chess.STARTING_FEN),
        ("After e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("After e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        ("Scholar's Mate", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
        ("Endgame - King and Pawn", "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"),
        ("Complex middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"),
        ("Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Queen's Gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2"),
        ("Winning endgame for White", "8/5k2/8/8/8/8/4PKP1/8 w - - 0 1"),
        ("Winning endgame for Black", "8/4pkp1/8/8/8/8/5K2/8 b - - 0 1"),
    ]

    print("\n" + "=" * 70)
    print("Testing predefined positions")
    print("=" * 70)

    results = []
    for name, fen in test_positions:
        print(f"\n[{name}]")
        board = chess.Board(fen)
        print(board)
        print(f"FEN: {fen}")

        eval_score = inference.evaluate_board(board)
        print(f"Evaluation: {format_evaluation(eval_score)}")
        print("-" * 70)

        results.append((name, eval_score))

    # Summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Position':<35} {'Eval (tanh)':<15} {'Centipawns':<15}")
    print("-" * 70)
    for name, score in results:
        clipped = np.clip(score, -0.9999, 0.9999)
        cp = np.arctanh(clipped) * 400
        print(f"{name:<35} {score:+.4f}         {cp:+.0f}")
    print("=" * 70)


def print_usage():
    """Print usage information"""
    print("\nUsage: python test_halfkp.py [OPTIONS] [WEIGHTS_FILE]")
    print("\nOptions:")
    print("  --test        Run predefined position tests")
    print("  --summary     Print model summary only (no interactive mode)")
    print("  --help, -h    Show this help message")
    print("\nExamples:")
    print("  python test_halfkp.py                    # Interactive mode with auto-detected weights")
    print("  python test_halfkp.py best_model.pt     # Interactive mode with specific .pt file")
    print("  python test_halfkp.py --test            # Run predefined tests")
    print("  python test_halfkp.py --summary model.pt  # Print model summary only")


if __name__ == "__main__":
    weights_file = None
    mode = "interactive"

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ['--help', '-h']:
            print_usage()
            sys.exit(0)
        elif arg == '--test':
            mode = "test"
        elif arg == '--summary':
            mode = "summary"
        elif not arg.startswith('-'):
            weights_file = arg
        i += 1

    if mode == "test":
        test_predefined_positions(weights_file)
    elif mode == "summary":
        try:
            inference = load_inference(weights_file)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        test_nnue_interactive(weights_file)
