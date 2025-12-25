import sys
import chess

# ------------------------------------
# Your engine import / implementation
# ------------------------------------
from best_move import find_best_move, MAX_NEGAMAX_DEPTH   # or inline your code


ENGINE_NAME = "DNN-Engine"
ENGINE_AUTHOR = "EK"

# ------------------------------------
# UCI Engine State
# ------------------------------------

board = chess.Board()
search_depth = MAX_NEGAMAX_DEPTH

# ------------------------------------
# Helpers
# ------------------------------------

def uci_print(msg):
    print(msg)
    sys.stdout.flush()

def set_position(tokens):
    global board
    idx = 0

    if tokens[idx] == "startpos":
        board = chess.Board()
        idx += 1
    elif tokens[idx] == "fen":
        fen = " ".join(tokens[idx + 1:idx + 7])
        board = chess.Board(fen)
        idx += 7

    if idx < len(tokens) and tokens[idx] == "moves":
        for move in tokens[idx + 1:]:
            board.push_uci(move)

# ------------------------------------
# Main UCI Loop
# ------------------------------------

def uci_loop():
    global search_depth

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            uci_print(f"id name {ENGINE_NAME}")
            uci_print(f"id author {ENGINE_AUTHOR}")
            uci_print("uciok")

        elif command == "isready":
            uci_print("readyok")

        elif command == "ucinewgame":
            board.reset()

        elif command == "position":
            set_position(tokens[1:])

        elif command == "go":
            # Optional: parse depth
            if "depth" in tokens:
                idx = tokens.index("depth")
                search_depth = int(tokens[idx + 1])

            fen = board.fen()
            best_move, score = find_best_move(fen, search_depth)

            # Convert move to UCI string if needed
            if isinstance(best_move, chess.Move):
                best_move_uci = best_move.uci()
            else:
                best_move_uci = best_move

            # Send info (optional but recommended)
            uci_print(f"info depth {search_depth} score cp {score}")

            uci_print(f"bestmove {best_move_uci}")
            board.push_uci(best_move_uci)

        elif command == "quit":
            break

        # Optional commands (safe to ignore)
        elif command in ("stop", "ponderhit"):
            pass

# ------------------------------------
# Entry Point
# ------------------------------------

if __name__ == "__main__":
    uci_loop()
