#!/usr/bin/env python3
import chess

from engine import find_best_move, MAX_NEGAMAX_DEPTH


class TimeCoontrol:
    pass


def uci_loop():
    board = chess.Board()

    while True:
        try:
            command = input().strip()
        except EOFError:
            break

        if command == "uci":
            print("id name DNN Engine")
            print("id author Eapen Kuruvilla")
            print("uciok", flush=True)

        elif command == "isready":
            print("readyok", flush=True)

        elif command.startswith("position"):
            tokens = command.split()

            if tokens[1] == "startpos":
                board.reset()
                move_index = 2


            elif tokens[1] == "fen":
                # FEN consists of the next 6 tokens AFTER "fen"
                fen = " ".join(tokens[2:8])
                board.set_fen(fen)
                move_index = 8

            else:
                raise ValueError(f"Unknown position format: {command}")

            # Apply moves if present
            if move_index < len(tokens) and tokens[move_index] == "moves":
                for mv in tokens[move_index + 1:]:
                    board.push_uci(mv)

        elif command.startswith("go"):
            tokens = command.split()
            movetime = None

            if "movetime" in tokens:
                movetime = int(tokens[tokens.index("movetime") + 1]) / 1000.0
            elif "wtime" in tokens and "btime" in tokens:
                wtime = int(tokens[tokens.index("wtime") + 1])
                btime = int(tokens[tokens.index("btime") + 1])
                time_left = wtime if board.turn else btime
                movetime = max(0.01, time_left / 30 / 1000)

            best_move, _ = find_best_move(
                board.fen(),
                max_depth=MAX_NEGAMAX_DEPTH,
                time_limit=movetime
            )

            print(f"bestmove {best_move.uci()}", flush=True)

        elif command == "stop":
            TimeCoontrol.stop_search = True

        elif command == "quit":
            break


if __name__ == "__main__":
    uci_loop()
