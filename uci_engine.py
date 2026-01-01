#!/usr/bin/env python3
import os
import sys
import threading

from engine import find_best_move, MAX_NEGAMAX_DEPTH, TimeControl
from engine import transposition_table, history_heuristic, killer_moves, qs_transposition_table

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
            transposition_table.clear()
            qs_transposition_table.clear()
            history_heuristic.clear()
            for i in range(len(killer_moves)):
                killer_moves[i] = [None, None]

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

            if "infinite" in tokens:
                movetime = None
            elif "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])
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
                #pv_str = " ".join(m.uci() for m in pv) if pv else best_move.uci()
                #print(f"info depth {max_depth} score cp {score} pv {pv_str}", flush=True)
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
            import chess

            f.write(f"chess: {chess.__file__}\n")
        except ImportError as e:
            f.write(f"chess import error: {e}\n")

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