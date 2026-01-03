import io
import os
import queue
import sys
import multiprocessing as mp
from typing import Iterable, Tuple, List
from collections import Counter

import chess.pgn
import numpy as np
import tensorflow as tf
import zstandard as zstd

# Board representation: 12 planes from side-to-move perspective
# Planes 0-5: "Our" pieces (side to move)
# Planes 6-11: "Their" pieces (opponent)
# Board flipped vertically when Black to move
  #TODO read/processing in parallel or divide the file among all the workers
BOARD_SHAPE = (12, 8, 8)

SHARD_SIZE = 1000_000  # samples per file
MAX_SHARDS = 1000
COMPRESSION = "GZIP"
OUT_DIR = "tfrecords"
TANH_SCALE = 400  # 1200(CP) = 3 = ~pi = 0.99
MAX_SCORE = 10 * TANH_SCALE  # tanh(10) is almost 1
MATE_FACTOR = 100
MAX_MATE_DEPTH = 5
MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR
MAX_MATERIAL_IMBALANCE = 250  # Skip positions with material imbalance > 250 cp

# Multiprocessing settings
NUM_WORKERS = min(5, max(1, mp.cpu_count() - 1))  # Leave one core free, cap at 5
BATCH_SIZE = 100  # Games per batch sent to workers
QUEUE_MAX_SIZE = 1000  # Max items in result queue

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Faster lookup by index (piece_type is 1-6)
_PIECE_TO_PLANE_LIST = [0, 0, 1, 2, 3, 4, 5]  # Index 0 unused

# Plane to piece character mapping
_PLANE_TO_PIECE_CHAR = ['p', 'n', 'b', 'r', 'q', 'k']  # Planes 0-5


def board_repr_to_fen(board_repr: np.ndarray, side_to_move: bool = chess.WHITE) -> str:
    """
    Convert a board representation back to FEN string.

    The board representation is from the side-to-move perspective:
    - Planes 0-5: "Our" pieces (side to move)
    - Planes 6-11: "Their" pieces (opponent)
    - Board is flipped vertically when Black to move

    Args:
        board_repr: (12, 8, 8) numpy array with piece planes
        side_to_move: chess.WHITE (True) or chess.BLACK (False), default WHITE

    Returns:
        FEN string with no castling rights, no en passant, halfmove=0, fullmove=1

    Example:
        >>> repr = get_board_repr_and_material(chess.Board())[0]
        >>> fen = board_repr_to_fen(repr, chess.WHITE)
        >>> print(fen)
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'
    """
    if board_repr.shape != BOARD_SHAPE:
        raise ValueError(f"Expected shape {BOARD_SHAPE}, got {board_repr.shape}")

    # Determine if we need to flip the board back
    # The representation is from side-to-move's perspective
    # If side_to_move is BLACK, the board was flipped, so we need to flip it back
    flip = not side_to_move

    # Build the board string rank by rank (from rank 8 to rank 1)
    ranks = []

    for row in range(8):  # row 0 = rank 8 in representation
        rank_str = ""
        empty_count = 0

        for col in range(8):
            # Get the actual row after considering flip
            actual_row = (7 - row) if flip else row

            piece_char = None

            # Check "our" pieces (planes 0-5) - these belong to side_to_move
            for plane in range(6):
                if board_repr[plane, actual_row, col] == 1:
                    char = _PLANE_TO_PIECE_CHAR[plane]
                    # "Our" pieces belong to side_to_move
                    if side_to_move == chess.WHITE:
                        piece_char = char.upper()  # White pieces uppercase
                    else:
                        piece_char = char.lower()  # Black pieces lowercase
                    break

            # Check "their" pieces (planes 6-11) - these belong to opponent
            if piece_char is None:
                for plane in range(6, 12):
                    if board_repr[plane, actual_row, col] == 1:
                        char = _PLANE_TO_PIECE_CHAR[plane - 6]
                        # "Their" pieces belong to opponent
                        if side_to_move == chess.WHITE:
                            piece_char = char.lower()  # Black pieces lowercase
                        else:
                            piece_char = char.upper()  # White pieces uppercase
                        break

            if piece_char is not None:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece_char
            else:
                empty_count += 1

        # Handle trailing empty squares
        if empty_count > 0:
            rank_str += str(empty_count)

        ranks.append(rank_str)

    # Construct FEN
    board_fen = "/".join(ranks)
    turn = "w" if side_to_move else "b"
    castling = "-"  # No castling rights as specified
    en_passant = "-"
    halfmove = "0"
    fullmove = "1"

    return f"{board_fen} {turn} {castling} {en_passant} {halfmove} {fullmove}"


def get_board_repr_and_material(board: chess.Board) -> Tuple[np.ndarray, int]:
    """
    Get board representation from side-to-move perspective and material imbalance.

    Returns:
        Tuple of:
        - (12, 8, 8) array with piece planes
        - Absolute material imbalance in centipawns
    """
    planes = np.zeros(BOARD_SHAPE, dtype=np.uint8)
    flip = not board.turn  # Flip if Black to move
    white_material = 0
    black_material = 0

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8

        if flip:
            row = 7 - row

        base = 0 if piece.color == board.turn else 6
        planes[base + _PIECE_TO_PLANE_LIST[piece.piece_type], row, col] = 1

        # Accumulate material
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value

    return planes, abs(white_material - black_material)


def is_quiet_position(board: chess.Board) -> bool:
    """
    Check if a position is quiet (suitable for static evaluation).

    A position is quiet when:
    - Not in check
    - No captures available
    - No checks available
    - No promotions available
    """
    if board.is_check():
        return False

    # Fast check: pawn on 7th rank = possible promotion
    our_pawns = board.pawns & board.occupied_co[board.turn]
    promotion_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
    if our_pawns & promotion_rank:
        return False

    for move in board.legal_moves:
        if board.is_capture(move):
            return False
        if board.gives_check(move):
            return False

    return True


def is_drawn_position(board: chess.Board) -> bool:
    """
    Check if a position is a draw or game is over.

    Returns True if:
    - Game is over (checkmate, stalemate, insufficient material,
      fivefold repetition, seventy-five move rule)
    - Draw can be claimed (threefold repetition, fifty-move rule)
    """
    return board.is_game_over() or board.can_claim_draw()


# ----- TF Example Helpers -----
def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def serialize(board: np.ndarray, score: float) -> bytes:
    feature = {
        "board": _bytes_feature(board.tobytes()),
        "score": _float_feature(score)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def process_game_pgn(pgn_text: str) -> Tuple[List[Tuple[np.ndarray, float]], dict]:
    """
    Process a single game from PGN text.

    Returns:
        Tuple of (list of (board_repr, score) tuples, kpi_updates dict)
    """
    kpi = Counter()
    results = []

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return results, kpi

        kpi["games"] += 1

        # Skip variant games
        if any("Variant" in key for key in game.headers.keys()):
            kpi["games_variant"] += 1
            return results, kpi

        node = game
        while (node := node.next()) is not None:
            if node.eval() is None:
                kpi["games_no_eval"] += 1
                break

            kpi["moves"] += 1
            board = node.board()

            # Check draw conditions first (cheaper than is_quiet_position)
            if is_drawn_position(board):
                kpi["moves_drawn"] += 1
                continue

            if not is_quiet_position(board):
                kpi["moves_not_quiet"] += 1
                continue

            # Get board representation and material imbalance in one pass
            board_repr, material_imbalance = get_board_repr_and_material(board)

            # Skip positions with large material imbalance
            if material_imbalance > MAX_MATERIAL_IMBALANCE:
                kpi["moves_material_imbalance"] += 1
                continue

            ev = node.eval()
            if ev.is_mate():
                kpi["moves_mate"] += 1
                mate_in = ev.white().mate()
                if mate_in < 0:
                    mate_in = max(-MAX_MATE_DEPTH, mate_in)
                    score = -MAX_SCORE - mate_in * MATE_FACTOR
                else:
                    mate_in = min(MAX_MATE_DEPTH, mate_in)
                    score = MAX_SCORE - mate_in * MATE_FACTOR
            else:
                score = ev.white().score()
                if abs(score) > MAX_NON_MATE_SCORE:
                    kpi["moves_score_too_large"] += 1
                    continue

            # Convert to side-to-move perspective
            if not board.turn:
                score = -score

            results.append((board_repr, score))

    except Exception as e:
        kpi["errors"] += 1

    return results, kpi


def worker_process(input_queue: mp.Queue, output_queue: mp.Queue, worker_id: int):
    """
    Worker process that processes games from input queue.
    """
    while True:
        try:
            item = input_queue.get()
            if item is None:  # Poison pill
                break

            batch_id, pgn_texts = item
            batch_results = []
            batch_kpi = Counter()

            for pgn_text in pgn_texts:
                results, kpi = process_game_pgn(pgn_text)
                batch_results.extend(results)
                batch_kpi.update(kpi)

            output_queue.put((batch_id, batch_results, batch_kpi))

        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            continue


def read_games_from_pgn_zst(path: str) -> Iterable[str]:
    """
    Read individual game PGN strings from a zstd-compressed PGN file.
    """
    dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)

    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

            current_game_lines = []
            in_game = False

            for line in text_stream:
                # Detect start of new game (starts with [Event or similar header)
                if line.startswith('['):
                    if in_game and current_game_lines:
                        # Yield previous game
                        yield ''.join(current_game_lines)
                        current_game_lines = []
                    in_game = True

                if in_game:
                    current_game_lines.append(line)

                    # Detect end of game (result token or empty line after moves)
                    if line.strip() in ('1-0', '0-1', '1/2-1/2', '*'):
                        yield ''.join(current_game_lines)
                        current_game_lines = []
                        in_game = False

            # Yield last game if exists
            if current_game_lines:
                yield ''.join(current_game_lines)


def stream_data_multiprocess(path: str, num_workers: int = NUM_WORKERS):
    """
    Stream training data using multiple worker processes.
    """
    print(f"Starting {num_workers} worker processes...")

    input_queue = mp.Queue(maxsize=num_workers * 2)
    output_queue = mp.Queue(maxsize=QUEUE_MAX_SIZE)

    # Start workers
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(input_queue, output_queue, i))
        p.start()
        workers.append(p)

    # Producer: read games and batch them
    total_kpi = Counter()
    batch = []
    batch_id = 0
    batches_sent = 0
    batches_received = 0

    # Start reading games
    game_reader = read_games_from_pgn_zst(path)
    reader_exhausted = False

    try:
        while True:
            # Send batches to workers (non-blocking when possible)
            while not reader_exhausted and not input_queue.full():
                try:
                    pgn_text = next(game_reader)
                    batch.append(pgn_text)

                    if len(batch) >= BATCH_SIZE:
                        input_queue.put((batch_id, batch))
                        batch_id += 1
                        batches_sent += 1
                        batch = []
                except StopIteration:
                    reader_exhausted = True
                    # Send remaining batch
                    if batch:
                        input_queue.put((batch_id, batch))
                        batches_sent += 1
                        batch = []
                    break
            # TODO read next batch before start waiting
            # Collect results from workers
            while True:
                try:
                    result_batch_id, results, kpi = output_queue.get(timeout=0.1)
                    batches_received += 1
                    total_kpi.update(kpi)

                    for board_repr, score in results:
                        yield board_repr, score

                    if batches_received % 10000 == 0:
                        print(f"Processed {batches_received}/{batches_sent} batches, {dict(total_kpi)}")

                except queue.Empty:
                    break

            # Check if done
            if reader_exhausted and batches_received >= batches_sent:
                break

    finally:
        # Send poison pills to workers
        for _ in workers:
            input_queue.put(None)

        # Wait for workers to finish
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        print(f"Final KPI: {dict(total_kpi)}")


def stream_data_single_process(path: str):
    """
    Stream training data using a single process (original implementation).
    """
    kpi = Counter()
    dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)

    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                kpi["games"] += 1

                if any("Variant" in key for key in game.headers.keys()):
                    kpi["games_variant"] += 1
                    continue

                node = game
                while (node := node.next()) is not None:
                    if node.eval() is None:
                        kpi["games_no_eval"] += 1
                        break

                    kpi["moves"] += 1
                    board = node.board()

                    # Check draw conditions first (cheaper than is_quiet_position)
                    if is_drawn_position(board):
                        kpi["moves_drawn"] += 1
                        continue

                    if not is_quiet_position(board):
                        kpi["moves_not_quiet"] += 1
                        continue

                    # Get board representation and material imbalance in one pass
                    board_repr, material_imbalance = get_board_repr_and_material(board)

                    # Skip positions with large material imbalance
                    if material_imbalance > MAX_MATERIAL_IMBALANCE:
                        kpi["moves_material_imbalance"] += 1
                        continue

                    ev = node.eval()
                    if ev.is_mate():
                        kpi["moves_mate"] += 1
                        mate_in = ev.white().mate()
                        if mate_in < 0:
                            mate_in = max(-MAX_MATE_DEPTH, mate_in)
                            score = -MAX_SCORE - mate_in * MATE_FACTOR
                        else:
                            mate_in = min(MAX_MATE_DEPTH, mate_in)
                            score = MAX_SCORE - mate_in * MATE_FACTOR
                    else:
                        score = ev.white().score()
                        if abs(score) > MAX_NON_MATE_SCORE:
                            kpi["moves_score_too_large"] += 1
                            continue

                    if not board.turn:
                        score = -score

                    yield board_repr, score

                if kpi["games"] % 10000 == 0:
                    print(f"Processed {kpi['games']} games...")

    print(f"Final KPI: {dict(kpi)}")


# ----- WRITE SHARDED TFRECORDS -----
def write_chess_tfrecords(
        data_stream: Iterable[Tuple[np.ndarray, float]],
        out_dir: str = OUT_DIR,
        shard_size: int = SHARD_SIZE) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    shard_id = 0
    samples_written = 0
    total_samples = 0
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
        total_samples += 1

        if samples_written >= shard_size:
            writer.close()
            print(f"Closed shard {shard_id} ({total_samples} total samples)")
            shard_id += 1
            samples_written = 0
            writer = None

        if shard_id >= MAX_SHARDS:
            break

    if writer:
        writer.close()
        print(f"Closed final shard {shard_id} ({total_samples} total samples)")

    return file_paths


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare chess training data from PGN files')
    parser.add_argument('input_file', help='Input PGN file (.pgn.zst)')
    parser.add_argument('--single-process', '-s', action='store_true',
                        help='Use single process mode (slower but uses less memory)')
    parser.add_argument('--workers', '-w', type=int, default=NUM_WORKERS,
                        help=f'Number of worker processes (default: {NUM_WORKERS})')
    parser.add_argument('--output-dir', '-o', default=OUT_DIR,
                        help=f'Output directory for TFRecords (default: {OUT_DIR})')

    args = parser.parse_args()

    if not args.input_file.endswith(".zst"):
        print('Error: A .pgn.zst file is expected as input')
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f'Error: File not found: {args.input_file}')
        sys.exit(1)

    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    if args.single_process:
        print("Using single-process mode")
        data_stream = stream_data_single_process(args.input_file)
    else:
        print(f"Using multi-process mode with {args.workers} workers")
        data_stream = stream_data_multiprocess(args.input_file, args.workers)

    write_chess_tfrecords(data_stream, out_dir=args.output_dir)
    print("Done!")


if __name__ == '__main__':
    main()