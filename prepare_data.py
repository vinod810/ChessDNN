import os
import tensorflow as tf
import numpy as np
from typing import Iterable, Tuple, List
import zstandard as zstd
import chess.pgn
import io

# ----- CONFIG -----
PGN_PATH = "pgn/lichess_db_standard_rated_2025-10.pgn.zst"
BOARD_SHAPE = (18, 8, 8)         # Example: 13 planes (12 piece planes + metadata)
#BOARD_DTYPE = np.uint8
#SCORE_DTYPE = np.float32
SHARD_SIZE = 1000_000 #100_000             # approx samples per file
MAX_SHARDS = 1000
COMPRESSION = "GZIP"             # reduces file size
OUT_DIR = "tfrecords"
MAX_SCORE = 10000 # http://talkchess.com/forum3/download/file.php?id=869
MATE_FACTOR = 100
MAX_NON_MATE_SCORE = MAX_SCORE - 20 * MATE_FACTOR


kpi = {
    "games": 0,
    "games_variant": 0,
    "games_no_eval": 0,
    "moves": 0,
    "moves_score_too_large": 0,
    "moves_material_change": 0,
    "moves_mate": 0
}



# def get_board_repr(board: chess.Board) :
#     """
#
#
#     Converts a chess.Board object into a PyTorch tensor suitable for DNN training.
#
#     Representation Scheme (18 Channels) - Absolute Orientation:
#     -----------------------------------------------------------
#     Channels 0-5:   White Pieces [P, N, B, R, Q, K]
#     Channels 6-11:  Black Pieces [P, N, B, R, Q, K]
#     Channel 12:     Turn (1 if White to move, 0 if Black)
#     Channel 13:     White King-side Castling Right
#     Channel 14:     White Queen-side Castling Right
#     Channel 15:     Black King-side Castling Right
#     Channel 16:     Black Queen-side Castling Right
#     Channel 17:     En-passant Target Square
#
#     Args:
#         board (chess.Board): The board state.
#
#     Returns:
#         torch.Tensor: A tensor of shape (18, 8, 8) containing the one-hot encoding.
#     """
#     # Map chess pieces to channel indices
#     PIECE_MAP = {
#         chess.PAWN: 0,
#         chess.KNIGHT: 1,
#         chess.BISHOP: 2,
#         chess.ROOK: 3,
#         chess.QUEEN: 4,
#         chess.KING: 5
#     }
#     # Initialize the tensor (Channels, Rows, Cols)
#     # 8x8 board, 18 channels
#     board_repr = np.zeros((18, 8, 8), dtype=BOARD_DTYPE)
#
#     # --- 1. FILL PIECE PLANES (0-11) ---
#     for square in chess.SQUARES:
#         piece = board.piece_at(square)
#         if piece:
#             # Get rank and file
#             rank = chess.square_rank(square)
#             file = chess.square_file(square)
#
#             # Determine channel offset: White=0, Black=6
#             channel_idx = PIECE_MAP[piece.piece_type]
#             if piece.color == chess.BLACK:
#                 channel_idx += 6
#
#             board_repr[channel_idx, rank, file] = 1
#
#     # --- 2. AUXILIARY PLANES (12-17) ---
#
#     # Channel 12: Turn (1 if White to move, 0 if Black)
#     if board.turn == chess.WHITE:
#         board_repr[12, :, :] = 1
#
#     # Castling Rights (Broadcast to entire plane)
#     # Indices are fixed in absolute mode
#     if board.has_kingside_castling_rights(chess.WHITE):
#         board_repr[13, :, :] = 1
#     if board.has_queenside_castling_rights(chess.WHITE):
#         board_repr[14, :, :] = 1
#     if board.has_kingside_castling_rights(chess.BLACK):
#         board_repr[15, :, :] = 1
#     if board.has_queenside_castling_rights(chess.BLACK):
#         board_repr[16, :, :] = 1
#
#     # Channel 17: En Passant Target
#     if board.ep_square is not None:
#         rank = chess.square_rank(board.ep_square)
#         file = chess.square_file(board.ep_square)
#         board_repr[17, rank, file] = 1
#
#     return board_repr

PIECE_TO_PLANE = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}


# def mobility_planes_64(board: chess.Board):
#     """
#     Returns mobility planes shape (64,8,8)
#     No transposing.
#     """
#     planes = np.zeros((64, 8, 8), dtype=np.float32)
#
#     for move in board.legal_moves:
#         s = move.from_square
#         t = move.to_square
#
#         dst_row = 7 - (t // 8)
#         dst_col = t % 8
#
#         planes[s, dst_row, dst_col] = 1.0
#
#     return planes


def get_board_repr(board: chess.Board):
    """
    Full 82-plane representation WITHOUT transpose.
    Output shape = (82, 8, 8)
    """

    # 18 base planes
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # -----------------------------
    # 1. Piece planes (12)
    # -----------------------------
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8

        base = 0 if piece.color == chess.WHITE else 6
        planes[base + PIECE_TO_PLANE[piece.piece_type], row, col] = 1.0

    # -----------------------------
    # 2. Side to move (1 plane)
    # -----------------------------
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # -----------------------------
    # 3. Castling rights (4 planes)
    # -----------------------------
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16] = 1.0

    # -----------------------------
    # 4. En-passant (1 plane)
    # -----------------------------
    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        file_idx = chess.square_file(board.ep_square)
        vec = np.zeros(8, dtype=np.float32)
        vec[file_idx] = 1.0
        ep_plane[:] = vec  # repeat 8-bit row across board

    planes[17] = ep_plane

    # -----------------------------
    # 5. Mobility planes (64)
    # -----------------------------
    # mob = mobility_planes_64(board)   # shape = (64,8,8)

    # -----------------------------
    # Combine: (18 + 64 = 82 planes)
    # -----------------------------
    # out = np.concatenate([planes, mob], axis=0)

    # return out   # shape = (82,8,8)
    return planes

# ----- TF Example Helpers -----
def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize(board: np.ndarray, score: float) -> bytes:
    """Convert a single (board, score) sample to serialized tf.Example."""
    feature = {
        "board": _bytes_feature(board.tobytes()),
        "score": _float_feature(score)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ----- WRITE SHARDED TFRECORDS -----
def write_chess_tfrecords(
    data_stream: Iterable[Tuple[np.ndarray, float]],
    out_dir: str = OUT_DIR,
    shard_size: int = SHARD_SIZE) -> List[str]:

    os.makedirs(out_dir, exist_ok=True)
    shard_id = 0
    samples_written = 0
    file_paths = []

    writer = None
    options = tf.io.TFRecordOptions(compression_type=COMPRESSION)

    for board, score in data_stream:

        if writer is None:
            path = os.path.join(out_dir, f"chess-{shard_id:05d}.tfrecord")
            writer = tf.io.TFRecordWriter(path, options=options)
            file_paths.append(path)
            print(f"Opened shard {shard_id}: {path}")

        writer.write(serialize(board, score))
        samples_written += 1

        if samples_written >= shard_size:
            writer.close()
            print(f"Closed shard {shard_id}")
            shard_id += 1
            samples_written = 0
            writer = None
            print(kpi)

        if shard_id >= MAX_SHARDS:
            break

    if writer:
        writer.close()
        print(f"Closed final shard {shard_id}")

    return file_paths

#def stream_chess_positions(n=500_000):
#    for _ in range(n):
#        board = np.random.randint(0, 2, size=BOARD_SHAPE, dtype=BOARD_DTYPE)
#        score = np.random.uniform(-10_000, 10_000)
#        yield board, score

def is_material_change(board: chess.Board) -> bool:
    """
    Returns True if the position is quiescent.
    A quiescent position has:
      - No checks
      - No captures
      - No promotions
      - No en-passant captures
    """
    # 1. If the side to move is in check, this is never quiescent
    # if board.is_check():
    #    return False

    # 2. Check all pseudo-legal moves to see if any are tactical (capture/promotions/ep)
    for move in board.generate_legal_moves():

        # (A) Captures, including en-passant
        if board.is_capture(move):
            return False

        # (B) Promotions
        if move.promotion is not None:
            return False

    # If no tactical moves, the position is quiet
    return True


def stream_data_from_pgn_zst(path):
    dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)

    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            # Convert binary decompressed stream → text stream
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
            #   yield game

                kpi["games"] += 1

                # Discard Variant games
                for key in game.headers.keys():
                    if "Variant" in key:
                        kpi["games_variant"] += 1
                        continue

                node = game
                while (node := node.next()) is not None:

                    if node.eval() is None:  # discard the game if no engine analysis available
                        kpi["games_no_eval"] += 1
                        break

                    kpi["moves"] += 1

                    if not is_material_change(node.board()):
                        kpi["moves_material_change"] += 1
                        continue

                    if node.eval().is_mate():
                        kpi["moves_mate"] += 1
                        score = node.eval().white().mate()
                        if score < 0:
                            score = -MAX_SCORE - score * MATE_FACTOR # score will be -ve if mate by black
                        else:
                            score = MAX_SCORE - score * MATE_FACTOR
                    else:
                        score = node.eval().white().score()
                        if abs(score) > MAX_NON_MATE_SCORE:  # Discard values larger than 90
                            kpi["moves_score_too_large"] += 1
                            continue

                    board_repr = get_board_repr(node.board())

                    yield board_repr, score

def main():
    write_chess_tfrecords(stream_data_from_pgn_zst(PGN_PATH))
    print(kpi)
    # Example:
    #i = 0
    #for game in stream_data_from_pgn_zst("pgn/lichess_db_standard_rated_2025-10.pgn.zst"):
    #    print(game.headers.get("Event"), game.headers.get("White"))
    #    i += 1
    #    if i > 5:
    #        exit()

if __name__ == '__main__':
    main()
    # write_chess_tfrecords(stream_chess_positions())
