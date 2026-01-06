import numpy as np
import chess
import sys
import os

# Import from the main training file
# Assumes the training code is saved as nnue_train.py
try:
    from halfkp import NNUEInference, NNUEFeatures
except ImportError:
    print("Error: Could not import from nnue_train.py")
    print("Make sure the training code is saved as 'nnue_train.py' in the same directory")
    sys.exit(1)


def format_evaluation(eval_score: float) -> str:
    """Format evaluation score for display"""
    # Convert tanh output back to centipawns
    clipped_score = np.clip(eval_score, -0.9999, 0.9999)
    cp = np.arctanh(clipped_score) * 400

    # Format output
    if abs(cp) < 10:
        return f"{eval_score:+.4f} (tanh) = {cp:+.1f} cp"
    else:
        return f"{eval_score:+.4f} (tanh) = {cp:+.0f} cp"


def display_board(board: chess.Board):
    """Display the chess board"""
    print("\n" + str(board))
    print(f"\nFEN: {board.fen()}")
    print(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")


def test_nnue_interactive():
    """Interactive testing interface for NNUE evaluation"""

    # Load the trained model
    weights_file = "nnue_weights.bin"

    if not os.path.exists(weights_file):
        print(f"Error: Weights file '{weights_file}' not found!")
        print("Please train the model first by running the main training script.")
        return

    print("Loading NNUE model...")
    try:
        inference = NNUEInference.load_weights(weights_file)
        print(f"Model loaded successfully from {weights_file}\n")
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
            # Get user input
            user_input = input("\nEnter FEN (or 'exit'/'quit'): ").strip()

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nExiting. Goodbye!")
                break

            # Check for empty input
            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['start', 'startpos', 'starting']:
                board = chess.Board()
            else:
                # Try to parse FEN
                try:
                    board = chess.Board(user_input)
                except ValueError as e:
                    print(f"\nError: Invalid FEN string!")
                    print(f"Details: {e}")
                    print("\nExample valid FEN:")
                    print("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                    continue

            # Display board
            display_board(board)

            # Check if position is valid
            if board.is_valid():
                # Extract features
                white_feat, black_feat = NNUEFeatures.board_to_features(board)

                # Evaluate position
                eval_score = inference.evaluate_board(board)

                # Display evaluation
                print(f"\nEvaluation: {format_evaluation(eval_score)}")

                # Additional info
                print(f"\nFeature counts:")
                print(f"  White perspective: {len(white_feat)} active features")
                print(f"  Black perspective: {len(black_feat)} active features")

                # Show interpretation
                if eval_score > 0.3:
                    print("Position favors White")
                elif eval_score < -0.3:
                    print("Position favors Black")
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


def test_predefined_positions():
    """Test with some predefined interesting positions"""

    weights_file = "nnue_weights.bin"
    if not os.path.exists(weights_file):
        print(f"Error: Weights file '{weights_file}' not found!")
        return

    print("Loading NNUE model...")
    inference = NNUEInference.load_weights(weights_file)

    test_positions = [
        ("Starting position", chess.STARTING_FEN),
        ("After e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("After e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        ("Scholar's Mate", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
        ("Endgame - King and Pawn", "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"),
        ("Complex middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"),
    ]

    print("\n" + "=" * 70)
    print("Testing predefined positions")
    print("=" * 70)

    for name, fen in test_positions:
        print(f"\n[{name}]")
        board = chess.Board(fen)
        print(board)
        print(f"FEN: {fen}")

        eval_score = inference.evaluate_board(board)
        print(f"Evaluation: {format_evaluation(eval_score)}")
        print("-" * 70)


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_predefined_positions()
    else:
        test_nnue_interactive()