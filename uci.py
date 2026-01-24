#!/usr/bin/env python3
import os
import threading
from pathlib import Path
import chess
import chess.polyglot

from engine import (find_best_move, MAX_NEGAMAX_DEPTH, TimeControl, dnn_eval_cache,
                    clear_game_history, game_position_history, HOME_DIR, kpi,
                    MAX_MP_CORES, IS_SHARED_TT_MP)
from book_move import init_opening_book, get_book_move
import mp_search

# Resign settings
RESIGN_THRESHOLD = -500  # centipawns
RESIGN_CONSECUTIVE_MOVES = 3  # must be losing for this many moves
resign_counter = 0

# Pondering settings
IS_PONDERING_ENABLED = True
PONDER_TIME_LIMIT = 600  # Maximum time for ponder search (safety cap)

CURR_DIR = Path(__file__).resolve().parent
DEFAULT_BOOK_PATH = CURR_DIR / f"../{HOME_DIR}" / 'book' / 'komodo.bin'

search_thread = None
use_book = True
use_ponder = True  # UCI Ponder option (can be disabled by GUI)

# Pondering state
is_pondering = False       # True when engine is in ponder mode
ponder_fen = None          # FEN of the position being pondered
ponder_hit_pending = False # True when ponderhit received, suppresses ponder output


# TODO support behind the screen pondering
# Don't generate any code yet. Just want to get your thoughts. How easy or difficult to add pondering in the
# background without relying on the UCI interface?  Once it is the engine's turn and if there is a ponder hit
# continue the evaluation for the remaining time.
# Soft ponder (easier): On ponder hit, stop the ponder search, then start a new search with remaining time.
# TT entries from pondering give you a head start—you'll blast through early depths quickly.
# 1. Engine makes move, records pv[1] as ponder_move
# 2. Spawn thread: search(ponder_position, time_limit=infinite)
# 3. Main thread: wait for opponent's move
# 4. Opponent moves:
#    - Set stop_search = True
#    - Wait for thread to finish
#    - If ponder hit: search from same position with real time limit
#      (TT is warm, will be fast)
#    - If ponder miss: search from new position with real time limit

def record_position_hash(board: chess.Board):
    """Record position in game history using Zobrist hash."""
    key = chess.polyglot.zobrist_hash(board)
    game_position_history[key] = game_position_history.get(key, 0) + 1


def uci_loop():
    global search_thread, use_book, use_ponder, RESIGN_THRESHOLD, RESIGN_CONSECUTIVE_MOVES, resign_counter
    global is_pondering, ponder_fen, ponder_hit_pending
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
            print(f"option name ResignThreshold type spin default {RESIGN_THRESHOLD} min -10000 max 0")
            print(f"option name ResignMoves type spin default {RESIGN_CONSECUTIVE_MOVES} min 1 max 10")
            print("option name Ponder type check default true")
            print(f"option name Threads type spin default {mp_search.MAX_MP_CORES} min 1 max 64")
            print(f"option name SharedTT type check default {'true' if mp_search.IS_SHARED_TT_MP else 'false'}")
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
                elif name.lower() == "resignthreshold":
                    RESIGN_THRESHOLD = int(value)
                elif name.lower() == "resignmoves":
                    RESIGN_CONSECUTIVE_MOVES = int(value)
                elif name.lower() == "ponder":
                    use_ponder = value.lower() == "true"
                elif name.lower() == "threads":
                    cores = int(value)
                    mp_search.set_mp_cores(cores)
                elif name.lower() == "sharedtt":
                    mp_search.set_shared_tt(value.lower() == "true")

        elif command == "ucinewgame":
            board.reset()
            dnn_eval_cache.clear()
            clear_game_history()
            resign_counter = 0
            is_pondering = False
            ponder_fen = None
            ponder_hit_pending = False
            mp_search.clear_shared_tables()  # Clear shared TT if MP enabled

        elif command.startswith("position"):
            # Stop any ongoing search (including ponder) before processing new position
            if search_thread and search_thread.is_alive():
                TimeControl.stop_search = True
                search_thread.join()
            is_pondering = False
            ponder_fen = None
            ponder_hit_pending = False

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
            go_ponder = "ponder" in tokens

            # If pondering is disabled or this is a ponder command but pondering not supported
            if go_ponder and (not IS_PONDERING_ENABLED or not use_ponder):
                # Ignore ponder request, just wait
                continue

            if "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])

            if go_ponder:
                # Ponder search - use safety time limit
                movetime = PONDER_TIME_LIMIT
                is_pondering = True
                ponder_fen = board.fen()
            elif "infinite" in tokens:
                movetime = None
            elif "movetime" in tokens:
                movetime = int(tokens[tokens.index("movetime") + 1]) / 1000.0
            elif "wtime" in tokens and "btime" in tokens:
                wtime = int(tokens[tokens.index("wtime") + 1])
                btime = int(tokens[tokens.index("btime") + 1])
                winc = int(tokens[tokens.index("winc") + 1]) if "winc" in tokens else 0
                binc = int(tokens[tokens.index("binc") + 1]) if "binc" in tokens else 0
                movestogo = int(tokens[tokens.index("movestogo") + 1]) if "movestogo" in tokens else 40

                time_left = wtime if board.turn else btime
                increment = winc if board.turn else binc

                # More conservative time management
                # Use smaller fraction of remaining time, larger safety buffer
                base_time = time_left / max(movestogo, 20)  # Don't divide by too few moves
                with_increment = base_time + increment * 0.7

                # Safety margins:
                # - At least 100ms buffer for overhead
                # - Never use more than 1/10th of remaining time in one move
                # - Keep at least 500ms in reserve
                max_for_move = (time_left - 500) / 10  # Never use more than 10% minus reserve

                movetime = min(with_increment, max_for_move) / 1000.0  # Convert to seconds
                movetime = max(0.1, movetime - 0.1)  # 100ms overhead buffer, 100ms minimum

            TimeControl.stop_search = False

            # Try book move first (but not when pondering - GUI already set up the position)
            book_move = None
            if use_book and not go_ponder:
                book_move = get_book_move(board, min_weight=1, temperature=1.0)

            if book_move:
                print(f"info string Book move: {book_move.uci()}", flush=True)
                # Try to get a ponder move from the book
                board.push(book_move)
                ponder_book_move = get_book_move(board, min_weight=1, temperature=1.0) if use_ponder else None
                board.pop()

                if ponder_book_move:
                    print(f"bestmove {book_move.uci()} ponder {ponder_book_move.uci()}", flush=True)
                else:
                    print(f"bestmove {book_move.uci()}", flush=True)
            else:
                fen = board.fen()

                def search_and_report():
                    global resign_counter, is_pondering

                    # Reset nodes counter before search
                    kpi['nodes'] = 0

                    # Use parallel search if MP enabled, otherwise single-threaded
                    if mp_search.is_mp_enabled():
                        best_move, score, pv, nodes, nps = mp_search.parallel_find_best_move(fen, max_depth=max_depth,
                                                                                             time_limit=movetime)
                        # Print final info line for MP search
                        if pv:
                            print(f"info depth {len(pv)} score cp {score} nodes {nodes} nps {nps} pv {' '.join(m.uci() for m in pv)}", flush=True)
                    else:
                        best_move, score, pv, _, _ = find_best_move(fen, max_depth=max_depth, time_limit=movetime)

                    # If ponderhit was received, suppress output (ponderhit search will output)
                    # But on regular stop (ponder miss), we MUST output bestmove
                    if ponder_hit_pending:
                        return

                    # Check for resign condition (not when pondering)
                    should_resign = False
                    if not is_pondering:
                        if score <= RESIGN_THRESHOLD:
                            resign_counter += 1
                            if resign_counter >= RESIGN_CONSECUTIVE_MOVES:
                                should_resign = True
                                print(f"info string Resigning (score {score} cp for {resign_counter} moves)", flush=True)
                        else:
                            resign_counter = 0

                    # Extract ponder move from PV if available
                    ponder_move = None
                    if use_ponder and len(pv) >= 2:
                        ponder_move = pv[1]

                    if best_move is None or best_move == chess.Move.null():
                        print("bestmove 0000", flush=True)
                    elif should_resign:
                        # Output bestmove with resign indication
                        # Some GUIs recognize "info string resign", others need manual handling
                        if ponder_move:
                            print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}", flush=True)
                        else:
                            print(f"bestmove {best_move.uci()}", flush=True)
                        print("info string resign", flush=True)
                    else:
                        if ponder_move:
                            print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}", flush=True)
                        else:
                            print(f"bestmove {best_move.uci()}", flush=True)

                    is_pondering = False

                search_thread = threading.Thread(target=search_and_report)
                search_thread.start()

        elif command == "ponderhit":
            # Ponder hit - the opponent played the move we were pondering on
            if is_pondering and search_thread and search_thread.is_alive():
                # Set flag to suppress ponder search output
                ponder_hit_pending = True

                # Stop the ponder search
                TimeControl.stop_search = True
                search_thread.join()

                # Reset flags
                ponder_hit_pending = False
                is_pondering = False

                # We need to recalculate time - but we don't have the time info here
                # UCI protocol expects the GUI to send a new "go" command after ponderhit
                # with the actual time constraints. Some GUIs do this, some don't.
                #
                # Standard behavior: just output the best move we found during pondering
                # The GUI should send a new "go" if it wants a fresh search.
                #
                # However, some engines continue searching. For simplicity, we'll
                # start a new search with a default time limit using the warm TT.

                if ponder_fen:
                    fen = ponder_fen

                    def ponderhit_search():
                        global resign_counter

                        kpi['nodes'] = 0

                        # Search with warm TT (clear_tt=False)
                        # Use a reasonable default time since we don't have clock info
                        # The GUI should ideally send new go command with time
                        if mp_search.is_mp_enabled():
                            best_move, score, pv, nodes, nps = mp_search.parallel_find_best_move(fen,
                                                                                                 max_depth=MAX_NEGAMAX_DEPTH,
                                                                                                 time_limit=10.0,
                                                                                                 clear_tt=False)
                            if pv:
                                print(f"info depth {len(pv)} score cp {score} nodes {nodes} nps {nps} pv {' '.join(m.uci() for m in pv)}", flush=True)
                        else:
                            best_move, score, pv, _, _ = find_best_move(fen, max_depth=MAX_NEGAMAX_DEPTH,
                                                                        time_limit=10.0, clear_tt=False)

                        # Check for resign condition
                        should_resign = False
                        if score <= RESIGN_THRESHOLD:
                            resign_counter += 1
                            if resign_counter >= RESIGN_CONSECUTIVE_MOVES:
                                should_resign = True
                                print(f"info string Resigning (score {score} cp for {resign_counter} moves)", flush=True)
                        else:
                            resign_counter = 0

                        # Extract ponder move from PV if available
                        ponder_move = None
                        if use_ponder and len(pv) >= 2:
                            ponder_move = pv[1]

                        if best_move is None or best_move == chess.Move.null():
                            print("bestmove 0000", flush=True)
                        elif should_resign:
                            if ponder_move:
                                print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}", flush=True)
                            else:
                                print(f"bestmove {best_move.uci()}", flush=True)
                            print("info string resign", flush=True)
                        else:
                            if ponder_move:
                                print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}", flush=True)
                            else:
                                print(f"bestmove {best_move.uci()}", flush=True)

                    TimeControl.stop_search = False
                    search_thread = threading.Thread(target=ponderhit_search)
                    search_thread.start()

        elif command == "stop":
            TimeControl.stop_search = True
            mp_search.stop_parallel_search()  # Signal MP workers to stop
            if search_thread and search_thread.is_alive():
                search_thread.join()
            # On ponder miss (stop during pondering), the search thread outputs bestmove
            # Reset all ponder state
            is_pondering = False
            ponder_hit_pending = False

        elif command == "quit":
            TimeControl.stop_search = True
            mp_search.stop_parallel_search()
            if search_thread and search_thread.is_alive():
                search_thread.join()
            mp_search.shutdown_worker_pool()  # Clean shutdown of workers
            break


if __name__ == "__main__":
    if os.path.exists(DEFAULT_BOOK_PATH):
        init_opening_book(str(DEFAULT_BOOK_PATH))
    else:
        print(f"info string Book not found: {DEFAULT_BOOK_PATH}", flush=True)

    uci_loop()