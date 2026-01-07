import numpy as np
import chess
import chess.pgn
import zstandard as zstd
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from typing import Tuple, List, Iterator, Optional
import struct
import random
import re

# Configuration
KING_SQUARES = 64
PIECE_SQUARES = 64
PIECE_TYPES = 5  # P, N, B, R, Q (no King)
COLORS = 2  # White, Black
INPUT_SIZE = KING_SQUARES * PIECE_SQUARES * PIECE_TYPES * COLORS
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1


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
        white_features, black_features: sparse feature indices
        stm: side to move (1.0 for white, 0.0 for black)
        """
        w_hidden = torch.clamp(self.ft(white_features), 0, 1)
        b_hidden = torch.clamp(self.ft(black_features), 0, 1)

        batch_size = white_features.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_size * 2, device=white_features.device)
        white_to_move = (stm > 0.5).squeeze()

        hidden[white_to_move, :self.hidden_size] = w_hidden[white_to_move]
        hidden[white_to_move, self.hidden_size:] = b_hidden[white_to_move]
        hidden[~white_to_move, :self.hidden_size] = b_hidden[~white_to_move]
        hidden[~white_to_move, self.hidden_size:] = w_hidden[~white_to_move]

        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return torch.tanh(x)


class StreamingChessDataset(IterableDataset):
    """Streaming dataset that reads evaluations from PGN comments"""

    def __init__(self, pgn_file: str,
                 max_positions_per_game: int = 20,
                 skip_early_moves: int = 10,
                 shuffle_buffer_size: int = 10000):
        self.pgn_file = pgn_file
        self.max_positions_per_game = max_positions_per_game
        self.skip_early_moves = skip_early_moves
        self.shuffle_buffer_size = shuffle_buffer_size

    @staticmethod
    def parse_evaluation(comment: str) -> Optional[float]:
        """
        Parse evaluation from PGN comment
        Supports formats:
        - [%eval 0.24] (centipawn score)
        - [%eval #3] (mate in 3)
        - [%eval -1.5] (negative score)
        - { [%eval 0.5] } (with curly braces)
        """
        if not comment:
            return None

        # Look for [%eval ...] pattern
        eval_pattern = r'\[%eval\s+(#)?(-?\d+\.?\d*)\]'
        match = re.search(eval_pattern, comment)

        if match:
            is_mate = match.group(1) == '#'
            value = float(match.group(2))

            if is_mate:
                # Mate score: convert to large centipawn value
                # Positive mate = white mates, negative = black mates
                cp = 10000 if value > 0 else -10000
            else:
                # Already in pawns, convert to centipawns
                cp = value * 100

            # Normalize using tanh(cp/400)
            return np.tanh(cp / 400.0)

        return None

    def process_game(self, game) -> Iterator[Tuple]:
        """Process a single game and yield positions with evaluations"""
        if game is None:
            return

        board = game.board()
        has_evaluations = False
        positions_yielded = 0

        # Iterate through the game tree
        node = game
        move_count = 0

        for node in game.mainline():
            move_count += 1

            if positions_yielded >= self.max_positions_per_game:
                break

            # Skip early moves
            if move_count <= self.skip_early_moves:
                board.push(node.move)
                continue

            # Make the move
            board.push(node.move)

            # Skip terminal positions
            if board.is_game_over():
                continue

            # Try to get evaluation from comment
            comment = node.comment if hasattr(node, 'comment') else ''
            eval_score = self.parse_evaluation(comment)

            if eval_score is not None:
                has_evaluations = True

                # Extract features
                white_feat, black_feat = NNUEFeatures.board_to_features(board)
                stm = 1.0 if board.turn == chess.WHITE else 0.0

                yield (white_feat, black_feat, stm, eval_score)
                positions_yielded += 1

        # Only log if we found evaluations
        if not has_evaluations and positions_yielded == 0:
            # Silently skip games without evaluations
            pass

    def stream_positions(self) -> Iterator[Tuple]:
        """Stream positions from PGN file"""
        try:
            with open(self.pgn_file, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')

                    games_processed = 0
                    games_with_evals = 0
                    positions_extracted = 0

                    while True:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            break

                        games_processed += 1

                        # Process game and count positions
                        position_count = 0
                        for position in self.process_game(game):
                            yield position
                            position_count += 1
                            positions_extracted += 1

                        if position_count > 0:
                            games_with_evals += 1

                        if games_processed % 1000 == 0:
                            eval_ratio = games_with_evals / games_processed * 100
                            print(f"Processed {games_processed} games: "
                                  f"{games_with_evals} with evals ({eval_ratio:.1f}%), "
                                  f"{positions_extracted} positions extracted")

        except FileNotFoundError:
            print(f"Error: File {self.pgn_file} not found!")
            print("Please provide a valid PGN file with evaluations.")
            return

    def __iter__(self) -> Iterator:
        """Return iterator with optional shuffling"""
        if self.shuffle_buffer_size > 0:
            return self.shuffled_iterator()
        else:
            return self.stream_positions()

    def shuffled_iterator(self) -> Iterator:
        """Iterator with shuffle buffer"""
        buffer = []
        for item in self.stream_positions():
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                while len(buffer) > self.shuffle_buffer_size // 2:
                    yield buffer.pop()

        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


def collate_fn(batch):
    """Custom collate function to handle sparse features"""
    white_inputs = []
    black_inputs = []
    stms = []
    scores = []

    for white_feat, black_feat, stm, score in batch:
        white_input = torch.zeros(INPUT_SIZE)
        black_input = torch.zeros(INPUT_SIZE)

        for f in white_feat:
            white_input[f] = 1.0
        for f in black_feat:
            black_input[f] = 1.0

        white_inputs.append(white_input)
        black_inputs.append(black_input)
        stms.append(stm)
        scores.append(score)

    return (torch.stack(white_inputs),
            torch.stack(black_inputs),
            torch.tensor(stms, dtype=torch.float32).unsqueeze(1),
            torch.tensor(scores, dtype=torch.float32).unsqueeze(1))


def train_model(model, train_loader, epochs=10, lr=0.001, positions_per_epoch=None):
    """Train the NNUE model with streaming data"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for batch_idx, (white_feat, black_feat, stm, target) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(white_feat, black_feat, stm)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")

            if positions_per_epoch and batch_count * train_loader.batch_size >= positions_per_epoch:
                break

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs} complete, Avg Loss: {avg_loss:.6f}")


class NNUEInference:
    """Numpy-based inference engine"""

    def __init__(self, model: NNUENetwork):
        """Initialize from PyTorch model"""
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
        """ClippedReLU activation: clip to [0, 1]"""
        return np.clip(x, 0, 1)

    def evaluate(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        """Evaluate position using numpy"""
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
        """Evaluate a chess board position"""
        white_feat, black_feat = NNUEFeatures.board_to_features(board)
        return self.evaluate(white_feat, black_feat, board.turn == chess.WHITE)

    def save_weights(self, filename: str):
        """Save weights to binary file"""
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
        """Load weights from binary file"""
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
    # Use PGN file with evaluations
    # Lichess PGN files downloaded with evaluations will have [%eval ...] comments
    pgn_file = "pgn/lichess_db_standard_rated_2025-10.pgn.zst"

    print("Creating streaming dataset from PGN evaluations...")
    print("Note: Only games with evaluation comments will be used")

    dataset = StreamingChessDataset(
        pgn_file,
        max_positions_per_game=20,
        skip_early_moves=10,
        shuffle_buffer_size=10000
    )

    train_loader = DataLoader(
        dataset,
        batch_size=256,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = NNUENetwork()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    train_model(
        model,
        train_loader,
        epochs=20,
        lr=0.001,
        positions_per_epoch=100000  # 100k positions per epoch
    )

    print("\nCreating numpy inference engine...")
    inference = NNUEInference(model)

    test_board = chess.Board()
    eval_score = inference.evaluate_board(test_board)
    print(f"Starting position evaluation: {eval_score:.4f}")
    print(f"Centipawn equivalent: {np.arctanh(np.clip(eval_score, -0.99, 0.99)) * 400:.1f}")

    inference.save_weights("nnue_weights.bin")
    print("Weights saved to nnue_weights.bin")