import os
import tensorflow as tf
import numpy as np
from typing import Iterable, Tuple, List
import zstandard as zstd
import chess.pgn
import io

# ----- CONFIG -----
PGN_PATH = "pgn/lichess_db_standard_rated_2025-10.pgn.zst"
BOARD_SHAPE = (13, 8, 8)     # TODO 13, 8, 8    # Example: 13 planes (12 piece planes + metadata)
SHARD_SIZE = 1000_000 #samples per file
MAX_SHARDS = 1000
COMPRESSION = "GZIP"
OUT_DIR = "tfrecords"
TANH_SCALE = 400 # 1200(CP) = 3 = ~pi = 0.99
MAX_SCORE = 10 * TANH_SCALE # tanh(10) is almost 1
 # http://talkchess.com/forum3/download/file.php?id=869
MATE_FACTOR = 100
MAX_MATE_DEPTH = 5
MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR

kpi = {
    "games": 0,
    "games_variant": 0,
    "games_no_eval": 0,
    "moves": 0,
    "moves_score_too_large": 0,
    "moves_capture": 0,
    "moves_mate": 0
}

PIECE_TO_PLANE = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def get_board_repr(board: chess.Board):
    planes = np.zeros(BOARD_SHAPE, dtype=np.uint8)

    # 1. Piece planes (12)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8

        base = 0 if piece.color == chess.WHITE else 6
        planes[base + PIECE_TO_PLANE[piece.piece_type], row, col] = 1

    # 2. Side to move (1 plane)
    planes[12, :, :] = 1 if board.turn == chess.WHITE else 0

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
            path = os.path.join(out_dir, f"chess-{shard_id:05d}.tfrecord.gz")
            writer = tf.io.TFRecordWriter(path, options=options)
            file_paths.append(path)
            print(f"Opened shard {shard_id}: {path}")

        writer.write(serialize(board, np.float32(score)))
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
        print(kpi)
        print(f"Closed final shard {shard_id}")

    return file_paths

def is_capture(board: chess.Board) -> bool:
    capture = (any(board.is_capture(mv) for mv in board.legal_moves))
    return capture


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

                    if is_capture(node.board()):
                        kpi["moves_capture"] += 1
                        continue

                    if node.eval().is_mate():
                        kpi["moves_mate"] += 1
                        score = node.eval().white().mate()
                        if score < 0:
                            score = min(-MAX_MATE_DEPTH, score) # score will be -ve if mate by black
                            score = -MAX_SCORE - score * MATE_FACTOR
                        else:
                            score = max(MAX_MATE_DEPTH, score) # score will be -ve if mate by black
                            score = MAX_SCORE - score * MATE_FACTOR
                    else:
                        score = node.eval().white().score()
                        if abs(score) > MAX_NON_MATE_SCORE:  # Discard values larger than 30 cp
                            kpi["moves_score_too_large"] += 1
                            continue

                    board_repr = get_board_repr(node.board())

                    yield board_repr, score

def main():
    write_chess_tfrecords(stream_data_from_pgn_zst(PGN_PATH))
    print(kpi)


if __name__ == '__main__':
    main()
