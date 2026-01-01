#!/usr/bin/env python3
import os
import sys
import threading
import chess  # âœ… Added missing import

from engine import find_best_move, MAX_NEGAMAX_DEPTH, TimeControl, dnn_eval_cache

search_thread = None


def uci_loop():
    global search_thread
    board = chess.Board()

    while True:
        try:
            command = input().strip()
        except EOFError:
            break

        if not command:
            continue

        if command == "uci":
            print("id name DNN Engine")
            print("id author Eapen Kuruvilla")
            print("uciok", flush=True)

        elif command == "isready":
            # Wait for any ongoing search to finish
            if search_thread and search_thread.is_alive():
                search_thread.join()
            print("readyok", flush=True)

        elif command == "ucinewgame":
            board.reset()
            dnn_eval_cache.clear()

        elif command.startswith("position"):
            tokens = command.split()
            if len(tokens) < 2:
                continue

            if tokens[1] == "startpos":
                board.reset()
                move_index = 2
            elif tokens[1] == "fen":
                fen = " ".join(tokens[2:8])
                board.set_fen(fen)
                move_index = 8
            else:
                continue

            if move_index < len(tokens) and tokens[move_index] == "moves":
                for mv in tokens[move_index + 1:]:
                    board.push_uci(mv)

        elif command.startswith("go"):
            tokens = command.split()
            movetime = None
            max_depth = MAX_NEGAMAX_DEPTH

            # âœ… Fixed: Parse depth separately (can coexist with time controls)
            if "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])

            # âœ… Fixed: Parse time controls (mutually exclusive with each other)
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
            fen = board.fen()

            def search_and_report():
                best_move, score, pv = find_best_move(fen, max_depth=max_depth, time_limit=movetime)
                if best_move is None:
                    print("bestmove 0000", flush=True)  # UCI null move
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
    # Log environment info to file for debugging
    with open("/tmp/uci_env_debug.log", "w") as f:
        f.write(f"Python: {sys.executable}\n")
        f.write(f"Version: {sys.version}\n")
        f.write(f"Path: {sys.path}\n")
        f.write(f"CWD: {os.getcwd()}\n")

        try:
            f.write(f"chess: {chess.__file__}\n")
        except Exception as e:
            f.write(f"chess error: {e}\n")

        try:
            import numpy
            f.write(f"numpy: {numpy.__file__}\n")
        except ImportError as e:
            f.write(f"numpy import error: {e}\n")

        try:
            import tensorflow
            f.write(f"tensorflow: {tensorflow.__file__}\n")
        except ImportError as e:
            f.write(f"tensorflow import error: {e}\n")

    uci_loop()