#!/usr/bin/env python3
import os
import threading
from pathlib import Path
import chess
import chess.polyglot

from engine import (find_best_move, MAX_NEGAMAX_DEPTH, TimeControl, dnn_eval_cache,
                    clear_game_history, game_position_history)
from book_move import init_opening_book, get_book_move

curr_dir = Path(__file__).resolve().parent
DEFAULT_BOOK_PATH = curr_dir / 'book' / 'komodo.bin'

search_thread = None
use_book = True

# TODO support behind the screen pondering

def record_position_hash(board: chess.Board):
    """Record position in game history using Zobrist hash."""
    key = chess.polyglot.zobrist_hash(board)
    game_position_history[key] = game_position_history.get(key, 0) + 1


def uci_loop():
    global search_thread, use_book
    board = chess.Board()
    book_path = DEFAULT_BOOK_PATH

    while True:
        try:
            command = input().strip()
        except EOFError:
            break

        if not command:
            continue

        if command == "uci":
            print("id name Neurofish")
            print("id author Eapen Kuruvilla")
            print("option name OwnBook type check default true")
            print("option name BookPath type string default " + str(DEFAULT_BOOK_PATH))
            print("uciok", flush=True)

        elif command == "isready":
            if search_thread and search_thread.is_alive():
                search_thread.join()
            print("readyok", flush=True)

        elif command.startswith("setoption"):
            tokens = command.split()
            if "name" in tokens and "value" in tokens:
                name_idx = tokens.index("name") + 1
                value_idx = tokens.index("value") + 1
                name = " ".join(tokens[name_idx:tokens.index("value")])
                value = " ".join(tokens[value_idx:])

                if name.lower() == "ownbook":
                    use_book = value.lower() == "true"
                elif name.lower() == "bookpath":
                    book_path = value
                    init_opening_book(book_path)

        elif command == "ucinewgame":
            board.reset()
            dnn_eval_cache.clear()
            clear_game_history()

        elif command.startswith("position"):
            tokens = command.split()
            if len(tokens) < 2:
                continue

            clear_game_history()

            if tokens[1] == "startpos":
                board.reset()
                move_index = 2
            elif tokens[1] == "fen":
                fen = " ".join(tokens[2:8])
                board.set_fen(fen)
                move_index = 8
            else:
                continue

            # Record starting position
            record_position_hash(board)

            # Apply moves and record each position
            if move_index < len(tokens) and tokens[move_index] == "moves":
                for mv in tokens[move_index + 1:]:
                    board.push_uci(mv)
                    record_position_hash(board)

        elif command.startswith("go"):
            tokens = command.split()
            movetime = None
            max_depth = MAX_NEGAMAX_DEPTH

            if "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])

            if "infinite" in tokens:
                movetime = None
            elif "movetime" in tokens:
                movetime = int(tokens[tokens.index("movetime") + 1]) / 1000.0
            elif "wtime" in tokens and "btime" in tokens:
                wtime = int(tokens[tokens.index("wtime") + 1])
                btime = int(tokens[tokens.index("btime") + 1])
                winc = int(tokens[tokens.index("winc") + 1]) if "winc" in tokens else 0
                binc = int(tokens[tokens.index("binc") + 1]) if "binc" in tokens else 0
                movestogo = int(tokens[tokens.index("movestogo") + 1]) if "movestogo" in tokens else 30

                time_left = wtime if board.turn else btime
                increment = winc if board.turn else binc
                movetime = max(0.05, (time_left / movestogo + increment * 0.8) / 1000 - 0.05)

            TimeControl.stop_search = False

            # Try book move first
            book_move = None
            if use_book:
                book_move = get_book_move(board, min_weight=1, temperature=1.0)

            if book_move:
                print(f"info string Book move: {book_move.uci()}", flush=True)
                print(f"bestmove {book_move.uci()}", flush=True)
            else:
                fen = board.fen()

                def search_and_report():
                    best_move, score, pv = find_best_move(fen, max_depth=max_depth, time_limit=movetime)
                    if best_move is None or best_move == chess.Move.null():
                        print("bestmove 0000", flush=True)
                    else:
                        print(f"bestmove {best_move.uci()}", flush=True)

                search_thread = threading.Thread(target=search_and_report)
                search_thread.start()

        elif command == "stop":
            TimeControl.stop_search = True
            if search_thread and search_thread.is_alive():
                search_thread.join()

        elif command == "quit":
            TimeControl.stop_search = True
            if search_thread and search_thread.is_alive():
                search_thread.join()
            break


if __name__ == "__main__":
    if os.path.exists(DEFAULT_BOOK_PATH):
        init_opening_book(str(DEFAULT_BOOK_PATH))
    else:
        print(f"info string Book not found: {DEFAULT_BOOK_PATH}", flush=True)

    uci_loop()