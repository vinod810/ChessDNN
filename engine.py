import importlib
import os
import random
import re
import time
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Optional
import chess
import chess.polyglot

from cached_board import CachedBoard
from nn_evaluator import DNNEvaluator, NNUEEvaluator, NNEvaluator
from nn_inference import MAX_SCORE

CURR_DIR = Path(__file__).resolve().parent

# # TODO Also support
# # SEE pruning+20-4
# # Futility pruning+15-30 Skip hopeless nodes
# # ## 6. Razoring to frontier nodes, if static eval is way below alpha, drop into quiescence.
# This could be +50-100 Elo but requires more experimentation.

# TODO Try NNE style accumulators of HalfKP

# TODO support multiprocessing
# # Run a search with OMP limited to 1 thread
# OMP_NUM_THREADS=1 python engine.py
# # Compare NPS to normal run
# # If NPS drops 3-4x → BLAS parallelism is huge → Lazy SMP less valuable +20-40 ELO
# # If NPS drops 1.5-2x → Search overhead dominates → Lazy SMP more valuable +50-80 ELO
#
# Lazy SMP Architecture Summary
# Process Structure
# MAIN PROCESS (UCI handler)
# ├── Handles UCI I/O (stdin/stdout)
# ├── Owns stop_event (multiprocessing.Event)
# ├── Owns work_queue, result_queue
# ├── Monitors time, sets stop_event when done
# └── Collects results, prints bestmove
#
# WORKER PROCESSES (N = cpu_count - 1)
# ├── Spawned once at engine startup
# ├── Each loads own DNN model copy
# ├── Shares TT, QS_TT, DNN cache via shared memory
# ├── Independent killer_moves, history_heuristic
# ├── Polls stop_event in search loop
# └── Reports completed depths to result_queue
# Start with Lazy SMP + independent caches first. Get that working (~60-80 Elo gain). Add shared TT later for the
# remaining ~20 Elo. This splits the complexity into two manageable steps.



HOME_DIR = "ChessDNN"
MIN_NEGAMAX_DEPTH = 3  # Minimum depth to complete regardless of time
MAX_NEGAMAX_DEPTH = 20
MAX_DEFAULT_TIME = 30
MAX_TABLE_SIZE = 200_000

IS_NUMPY_EVAL = True
IS_BLAS_ENABLED = False

IS_NN_ENABLED = True
NN_TYPE = "DNN"
DNN_MODEL_FILEPATH = CURR_DIR / f"../{HOME_DIR}" / 'model' / 'dnn.pt'
NNUE_MODEL_FILEPATH = CURR_DIR / f"../{HOME_DIR}" / 'model' / 'nnue.pt'

QS_DEPTH_MAX_NN_EVAL_UNCONDITIONAL = 1
QS_DEPTH_MAX_NN_EVAL_CONDITIONAL = 10
DELTA_MAX_NN_EVAL = 50  # Score difference, below which will trigger a NN evaluation
STAND_PAT_MAX_NN_EVAL = 200

QS_TT_SUPPORTED = True
DELTA_PRUNING_QS_MIN_DEPTH = 6
DELTA_PRUNING_MARGIN = 75
TACTICAL_QS_MAX_DEPTH = 5

ASPIRATION_WINDOW = 40
MAX_AW_RETRIES = 3

LMR_MOVE_THRESHOLD = 2
LMR_MIN_DEPTH = 3  # minimum depth to apply LMR

NULL_MOVE_REDUCTION = 3  # R value (usually 2 or 3)
NULL_MOVE_MIN_DEPTH = 4

SINGULAR_MARGIN = 130  # Score difference in centipawns to trigger singular extension
SINGULAR_EXTENSION = 1  # Extra depth

# Time management
ESTIMATED_BRANCHING_FACTOR = 2.5  # Typical branching factor after pruning
TIME_SAFETY_MARGIN = 0.7  # Only start new depth if we estimate having 70%+ of needed time


if not IS_BLAS_ENABLED:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

MODEL_PATH = str(DNN_MODEL_FILEPATH if NN_TYPE == "DNN" else NNUE_MODEL_FILEPATH)

class TimeControl:
    time_limit = None  # in seconds
    start_time = None
    stop_search = False  # Set by UCI 'stop' command - always honored
    soft_stop = False  # Set by time limit - ignored until MIN_DEPTH reached


TTEntry = namedtuple("TTEntry", ["depth", "score", "flag", "best_move"])
TT_EXACT, TT_LOWER_BOUND, TT_UPPER_BOUND = 0, 1, 2

transposition_table = {}
qs_transposition_table = {}
dnn_eval_cache = {}
killer_moves = [[None, None] for _ in range(MAX_NEGAMAX_DEPTH + 1)]
history_heuristic = {}

kpi = {
    "nodes": 0,
    "pos_eval": 0,
    "dnn_evals": 0,
    "beta_cutoffs": 0,
    "tt_hits": 0,
    "qs_tt_hits": 0,
    "dec_hits": 0,  # DNN evaluation cache hits
    "q_depth": 0,
}

# Track positions seen in the current game (cleared on ucinewgame)
game_position_history: dict[int, int] = {}  # zobrist_hash -> count

nn_evaluator: DNNEvaluator | NNUEEvaluator | None = None

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def clear_game_history():
    """Clear game position history (call on ucinewgame)."""
    game_position_history.clear()


def is_draw_by_repetition(board: CachedBoard) -> bool:
    """
    Check for threefold repetition combining game history and search path.

    A position is a draw if it occurs 3+ times total:
    - game_position_history counts occurrences in the actual game (including current position)
    - board.is_repetition() counts occurrences in the search tree

    Key insight: The ROOT position of search is already counted in game_history.
    If we return to that position during search, is_repetition(2) will be True,
    but we shouldn't double-count.

    Simple approach:
    - If game_count >= 3: already 3-fold in game alone
    - If game_count == 2: one more occurrence makes it 3-fold
      → is_repetition(2) means we've seen this position again in search
    - If game_count == 1: need 2 more occurrences
      → is_repetition(2) means we've seen this position 2+ times in current search line
    - If game_count == 0: need 3 occurrences in search alone
      → is_repetition(3)
    """
    key = board.zobrist_hash()
    game_count = game_position_history.get(key, 0)

    # Already occurred 3+ times in game
    if game_count >= 3:
        return True

    # Occurred twice in game - need to see it once more in search
    # is_repetition(2) = position occurred 2+ times including current
    if game_count == 2:
        # Any occurrence in search (beyond root) makes it 3-fold
        # is_repetition(2) catches: we're at this position AND we were here before in search
        if board.is_repetition(2):
            return True

    # Occurred once in game - need to see it twice more in search
    # is_repetition(3) = position occurred 3+ times including current
    if game_count == 1:
        if board.is_repetition(2):
            # We've seen it twice in search (including current), plus once in game = 3
            return True

    # Never in game - need 3 occurrences in search
    if game_count == 0:
        if board.is_repetition(3):
            return True

    return False


def check_time():
    """Check if time limit exceeded. Sets soft_stop flag."""
    if TimeControl.time_limit is None:
        return

    if (time.perf_counter() - TimeControl.start_time) >= TimeControl.time_limit:
        TimeControl.soft_stop = True


def evaluate_material(board: CachedBoard) -> int:
    """
    Evaluate the board position from the side-to-move perspective.

    Returns:
        Score in centipawns. Positive = good for side to move.
    """
    # Check for game over conditions
    if board.is_game_over():
        if board.is_checkmate():
            # Side to move is checkmated - worst possible score
            return -MAX_SCORE + board.ply()
        else:
            return 0  # Stalemate or other draw

    kpi['pos_eval'] += 1
    score = board.material_evaluation()
    return score


def evaluate_dnn(board: CachedBoard) -> int:
    if board.is_game_over():
        if board.is_checkmate():
            # Side to move is checkmated - worst possible score
            return -MAX_SCORE + board.ply()
        else:
            return 0  # Stalemate or other draw

    key = board.zobrist_hash()
    if key in dnn_eval_cache:
        kpi['dec_hits'] += 1
        return dnn_eval_cache[key]

    # assert (board.is_quiet_position())
    kpi['dnn_evals'] += 1
    score = nn_evaluator.evaluate_centipawns(board)
    dnn_eval_cache[key] = score

    # Occasionally do a full evaluation to rule out any drift errors.
    if kpi['dnn_evals'] % 10_000 == 0:
        full_score = nn_evaluator.evaluate_full_centipawns(board)
        if abs(full_score - score) > 10:
            print(f"Warning: incremental({score}))and full({full_score}) evaluation differ, {board.fen()}!")

    return score


def move_score_q_search(board: CachedBoard, move: chess.Move) -> int:
    """
    Score a move for quiescence search ordering.
    Uses cached move info for efficiency.
    """
    score = 0

    # Use cached is_capture and piece types
    if board.is_capture_cached(move):
        victim_type = board.get_victim_type(move)
        attacker_type = board.get_attacker_type(move)
        if victim_type and attacker_type:
            score += 10 * PIECE_VALUES[victim_type] - PIECE_VALUES[attacker_type]

    return score


def move_score(board: CachedBoard, move: chess.Move, depth: int) -> int:
    """
    Score a move for move ordering.
    Uses cached move info for efficiency.
    """
    score = 0

    # Use cached is_capture
    is_capture = board.is_capture_cached(move)

    # Killer moves (quiet moves only)
    if not is_capture and depth is not None and 0 <= depth < len(killer_moves):
        if move == killer_moves[depth][0]:
            score += 9000
        elif move == killer_moves[depth][1]:
            score += 8000

    if is_capture:
        # Use cached victim and attacker types
        victim_type = board.get_victim_type(move)
        attacker_type = board.get_attacker_type(move)
        if victim_type and attacker_type:
            score += 10 * PIECE_VALUES[victim_type] - PIECE_VALUES[attacker_type]

    score += history_heuristic.get(
        (move.from_square, move.to_square), 0
    )

    # Use cached gives_check
    if board.gives_check_cached(move):
        score += 50

    return score


def ordered_moves(board: CachedBoard, depth: int, pv_move=None, tt_move=None):
    """
    Return legal moves ordered by expected quality.
    Pre-computes move info for efficient scoring.
    """
    # Ensure move info is precomputed for this position
    board.precompute_move_info()

    scored_moves = []
    for move in board.get_legal_moves_list():
        if move == tt_move:
            score = 1000000
        elif move == pv_move:
            score = 900000
        else:
            score = move_score(board, move, depth)
        scored_moves.append((score, move))

    # Add small random noise to break ties (preserves relative ordering mostly)
    scored_moves.sort(key=lambda tup: (tup[0], random.random()), reverse=True)
    return [move for _, move in scored_moves]


def ordered_moves_q_search(board: CachedBoard):
    """
    Return legal moves ordered for quiescence search.
    Pre-computes move info for efficient scoring.
    """
    # Ensure move info is precomputed for this position
    board.precompute_move_info()

    # Use the already-cached legal moves list (don't create a copy)
    moves = board.get_legal_moves_list()
    moves_with_scores = [(move_score_q_search(board, m), m) for m in moves]
    moves_with_scores.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in moves_with_scores]


def should_stop_search(current_depth: int) -> bool:
    """
    Determine if search should stop.
    - TimeControl.stop_search (from UCI 'stop'): Always honored immediately
    - TimeControl.soft_stop (from time limit): Only honored after MIN_DEPTH
    """
    if TimeControl.stop_search:
        return True
    if TimeControl.soft_stop and current_depth > MIN_NEGAMAX_DEPTH:
        return True
    return False


def quiescence(board: CachedBoard, alpha: int, beta: int, q_depth: int) -> Tuple[int, List[chess.Move]]:
    """
    Quiescence search.

    Returns:
        Tuple of (score, pv) where pv is the list of moves in the principal variation.
    """
    kpi['q_depth'] = max(kpi['q_depth'], q_depth)
    kpi['nodes'] += 1

    check_time()
    if TimeControl.stop_search:
        raise TimeoutError()

    # -------- Draw detection --------
    if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
        return get_draw_score(board), []

    # -------- Always compute key for TT --------
    key = board.zobrist_hash()

    if QS_TT_SUPPORTED and key in qs_transposition_table:
        kpi['qs_tt_hits'] += 1
        stored_score = qs_transposition_table[key]
        if stored_score >= beta:
            return stored_score, []  # ✅ Return stored score, not beta

    # -------- Check for game over --------
    if board.is_game_over():
        if board.is_checkmate():
            return -MAX_SCORE + board.ply(), []
        return 0, []  # Stalemate or draw

    is_check = board.is_check()
    best_pv = []
    best_score = -MAX_SCORE  # ✅ Track best score separately

    # -------- Stand pat (not valid when in check) --------
    if not is_check:
        is_dnn_eval = False
        if IS_NN_ENABLED and q_depth <= QS_DEPTH_MAX_NN_EVAL_UNCONDITIONAL:
            stand_pat = evaluate_dnn(board)
            is_dnn_eval = True
        else:
            stand_pat = evaluate_material(board)

        if (not is_dnn_eval and IS_NN_ENABLED and q_depth <= QS_DEPTH_MAX_NN_EVAL_CONDITIONAL
                and abs(stand_pat) < STAND_PAT_MAX_NN_EVAL
                and abs(stand_pat - beta) < DELTA_MAX_NN_EVAL):
            stand_pat = evaluate_dnn(board)
            is_dnn_eval = True

        if stand_pat >= beta:
            kpi['beta_cutoffs'] += 1
            return stand_pat, []  # ✅ Return stand_pat, not beta

        # Big delta pruning - can't possibly reach alpha
        if stand_pat + PIECE_VALUES[chess.QUEEN] < alpha:
            return stand_pat, []  # ✅ Return stand_pat (it's the best we can do)

        if (not is_dnn_eval and IS_NN_ENABLED
                and q_depth <= QS_DEPTH_MAX_NN_EVAL_CONDITIONAL
                and abs(stand_pat) < STAND_PAT_MAX_NN_EVAL
                and (stand_pat > alpha or abs(stand_pat - alpha) < DELTA_MAX_NN_EVAL)):
            stand_pat = evaluate_dnn(board)

        best_score = stand_pat  # ✅ Initialize best_score with stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    moves_searched = 0

    # Pre-compute move info for this position
    board.precompute_move_info()

    for move in ordered_moves_q_search(board):
        if not is_check:
            # Use cached is_capture
            is_capture_move = board.is_capture_cached(move)
            # If the move gives check and depth < TACTICAL_QS_MAX_DEPTH, go ahead with
            # evaluation
            if not is_capture_move:
                if q_depth > TACTICAL_QS_MAX_DEPTH:
                    continue
                # Use cached gives_check
                if not board.gives_check_cached(move):
                    continue

            # -------- Delta pruning (only when not in check) --------
            if (q_depth >= DELTA_PRUNING_QS_MIN_DEPTH and is_capture_move
                    and not board.gives_check_cached(move)):
                victim_type = board.get_victim_type(move)
                if victim_type:
                    gain = PIECE_VALUES[victim_type]
                    if best_score + gain + DELTA_PRUNING_MARGIN < alpha:  # ✅ Use best_score
                        continue

        push_move(board, move, nn_evaluator)
        score, child_pv = quiescence(board, -beta, -alpha, q_depth + 1)
        score = -score
        board.pop()
        nn_evaluator.pop()
        moves_searched += 1

        if score > best_score:  # ✅ Track best_score
            best_score = score
            best_pv = [move] + child_pv

        if score >= beta:
            kpi['beta_cutoffs'] += 1
            if QS_TT_SUPPORTED:
                qs_transposition_table[key] = score
            return score, best_pv  # ✅ Return score, not beta

        if score > alpha:
            alpha = score

    # -------- No moves searched when in check = checkmate --------
    if is_check and moves_searched == 0:
        return -MAX_SCORE + board.ply(), []

    if QS_TT_SUPPORTED:
        qs_transposition_table[key] = best_score  # ✅ Store best_score

    return best_score, best_pv  # ✅ Return best_score, not alpha


def get_draw_score(board: CachedBoard) -> int:
    """
    Return score for draw positions with capped contempt.

    Uses -material (proportional contempt) but capped to avoid situations
    where we'd rather lose material than draw.

    When winning: contempt ranges from 0 to -300 (draw is bad)
    When losing: contempt ranges from 0 to +150 (draw is good)
    """
    material = board.material_evaluation()

    if material > 0:
        return max(-300, -material)
    elif material < 0:
        return min(150, -material)
    return 0


def negamax(board: CachedBoard, depth: int, alpha: int, beta: int, allow_singular: bool = True) -> Tuple[
    int, List[chess.Move]]:
    """
    Negamax search with alpha-beta pruning.

    Returns:
        Tuple of (score, pv) where pv is the list of moves in the principal variation.
    """
    kpi['nodes'] += 1

    check_time()
    if TimeControl.stop_search:
        raise TimeoutError()

    # -------- Draw detection --------
    if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
        # print(f"Eapen, get_draw_score(board)={get_draw_score(board)}, dpeth={depth}, turn={board.turn}")
        return get_draw_score(board), []

    key = board.zobrist_hash()
    alpha_orig = alpha
    beta_orig = beta  # ✅ FIX 1

    best_move = None
    best_pv = []
    max_eval = -MAX_SCORE

    # -------- TT Lookup --------
    entry = transposition_table.get(key)
    if entry and entry.depth >= depth:
        kpi['tt_hits'] += 1
        if entry.flag == TT_EXACT:
            return entry.score, extract_pv_from_tt(board, depth)
        elif entry.flag == TT_LOWER_BOUND:
            alpha = max(alpha, entry.score)
        elif entry.flag == TT_UPPER_BOUND:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score, []  # ✅ FIX 2 (no PV on cutoff)

        tt_move = entry.best_move
    else:
        tt_move = None

    # -------- Quiescence if depth == 0 --------
    if depth == 0:
        return quiescence(board, alpha, beta, 1)

    in_check = board.is_check()

    # -------- Null Move Pruning (not when in check or in zugzwang-prone positions) --------
    if (depth >= NULL_MOVE_MIN_DEPTH
            and not in_check
            # and not on_expected_pv  # Don't null-move on PV
            and board.has_non_pawn_material()
            and board.occupied.bit_count() > 6):
        push_move(board, chess.Move.null(), nn_evaluator)
        score, _ = negamax(board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, allow_singular=False)
        score = -score
        board.pop()
        nn_evaluator.pop()
        if score >= beta:
            return beta, []

    # -------- Singular Extension Check (simplified) --------
    singular_extension_applicable = False
    singular_move = None

    if (allow_singular
            and depth >= 6
            and tt_move is not None
            and not in_check
            and key in transposition_table
            and transposition_table[key].flag != TT_UPPER_BOUND
            and transposition_table[key].depth >= depth - 3):

        reduced_depth = max(1, depth // 2 - 1)
        reduced_beta = transposition_table[key].score - SINGULAR_MARGIN

        # Only search a few top moves, not all
        move_count = 0
        highest_score = -MAX_SCORE

        for move in ordered_moves(board, depth, tt_move=tt_move):
            if move == tt_move:
                continue
            if move_count >= 3:  # Limit moves checked
                break

            push_move(board, move, nn_evaluator)
            score, _ = negamax(board, reduced_depth, -reduced_beta - 1, -reduced_beta, allow_singular=False)
            score = -score
            board.pop()
            nn_evaluator.pop()
            move_count += 1

            highest_score = max(highest_score, score)
            if highest_score >= reduced_beta:
                break

        if highest_score < reduced_beta:
            singular_extension_applicable = True
            singular_move = tt_move

    # -------- Move Ordering --------
    moves = ordered_moves(board, depth, tt_move=tt_move)
    if not moves:
        if in_check:
            return -MAX_SCORE + board.ply(), []
        return 0, []

    for move_index, move in enumerate(moves):
        # Use cached move info
        is_capture = board.is_capture_cached(move)
        gives_check = board.gives_check_cached(move)

        push_move(board, move, nn_evaluator)
        child_in_check = board.is_check()

        # -------- Extensions --------
        extension = 0
        if singular_extension_applicable and move == singular_move:
            extension = SINGULAR_EXTENSION
        elif child_in_check:
            extension = 1

        new_depth = depth - 1 + extension

        # -------- LMR --------
        if depth is not None and 0 <= depth < len(killer_moves):
            km0 = killer_moves[depth][0]
            km1 = killer_moves[depth][1]
        else:
            km0 = None
            km1 = None

        reduce = (
                depth >= LMR_MIN_DEPTH
                and move_index >= LMR_MOVE_THRESHOLD
                and not child_in_check
                and not is_capture
                and not gives_check
                and move != km0
                and move != km1
                and extension == 0
        )

        if reduce:
            reduction = 1
            if depth >= 6 and move_index >= 6:
                reduction = 2
            score, child_pv = negamax(board, new_depth - reduction, -alpha - 1, -alpha, allow_singular=True)
            score = -score
            if score > alpha:
                score, child_pv = negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                score = -score
        else:
            if move_index > 0:
                score, child_pv = negamax(board, new_depth, -alpha - 1, -alpha, allow_singular=True)
                score = -score
                if alpha < score < beta:
                    score, child_pv = negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                    score = -score
            else:
                score, child_pv = negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                score = -score

        board.pop()
        nn_evaluator.pop()

        if score > max_eval:
            max_eval = score
            best_move = move
            best_pv = [move] + child_pv

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Update killer moves and history for quiet moves
            if not is_capture and depth is not None and 0 <= depth < len(killer_moves):
                if killer_moves[depth][0] != move:
                    killer_moves[depth][1] = killer_moves[depth][0]
                    killer_moves[depth][0] = move
                key_hist = (move.from_square, move.to_square)
                history_heuristic[key_hist] = min(
                    history_heuristic.get(key_hist, 0) + depth * depth, 10_000
                )
            kpi['beta_cutoffs'] += 1
            break

    # -------- Store in TT --------
    if max_eval <= alpha_orig:
        flag = TT_UPPER_BOUND
    elif max_eval >= beta_orig:  # ✅ FIX 1
        flag = TT_LOWER_BOUND
    else:
        flag = TT_EXACT

    old = transposition_table.get(key)
    if old is None or depth >= old.depth:
        transposition_table[key] = TTEntry(depth, max_eval, flag, best_move)

    return max_eval, best_pv


def extract_pv_from_tt(board: CachedBoard, max_depth: int) -> List[chess.Move]:
    """
    Extract the principal variation from the transposition table.

    Args:
        board: Current board position
        max_depth: Maximum depth to extract

    Returns:
        List of moves forming the PV
    """
    pv = []
    seen_keys = set()

    for _ in range(max_depth):
        key = board.zobrist_hash()

        # Prevent infinite loops from repetitions
        if key in seen_keys:
            break
        seen_keys.add(key)

        entry = transposition_table.get(key)
        if entry is None or entry.best_move is None:
            break

        move = entry.best_move
        if move not in board.get_legal_moves_list():
            break

        pv.append(move)
        push_move(board, move, nn_evaluator)

    # Restore board state
    for _ in range(len(pv)):
        board.pop()
        nn_evaluator.pop()

    return pv


def age_heuristic_history():
    for k in list(history_heuristic.keys()):
        history_heuristic[k] = history_heuristic[k] * 3 // 4


def control_dict_size(table, max_dict_size):
    if len(table) > max_dict_size:
        for _ in range(max_dict_size // 4):
            table.pop(next(iter(table)))


def pv_to_san(board: CachedBoard, pv: List[chess.Move]) -> str:
    """
    Convert a PV (list of moves) to SAN notation string.

    Args:
        board: Starting position
        pv: List of moves

    Returns:
        SAN string representation of the PV
    """
    san_moves = []
    temp_board = board.copy(stack=False)

    for i, move in enumerate(pv):
        if temp_board.turn == chess.WHITE:
            move_num = temp_board.fullmove_number
            san_moves.append(f"{move_num}.")
        elif i == 0:
            # Black to move at start
            move_num = temp_board.fullmove_number
            san_moves.append(f"{move_num}...")

        san_moves.append(temp_board.san(move))
        push_move(temp_board, move, nn_evaluator)

    return " ".join(san_moves)


def find_best_move(fen, max_depth=MAX_NEGAMAX_DEPTH, time_limit=None, expected_best_moves=None) -> Tuple[
    Optional[chess.Move], int, List[chess.Move]]:
    """
    Finds the best move for a given FEN using iterative deepening negamax with alpha-beta pruning,
    aspiration windows, TT, quiescence, null-move pruning, LMR, singular extensions, and heuristics.

    Returns:
        Tuple of (best_move, score, pv)
    """

    # -------- Initialize time control --------
    TimeControl.time_limit = time_limit
    TimeControl.stop_search = False
    TimeControl.soft_stop = False  # Reset soft stop
    TimeControl.start_time = time.perf_counter()

    # -------- Clear search tables & heuristics --------
    for i in range(len(killer_moves)):
        killer_moves[i] = [None, None]
    history_heuristic.clear()
    transposition_table.clear()
    qs_transposition_table.clear()

    control_dict_size(dnn_eval_cache, MAX_TABLE_SIZE)

    board = CachedBoard(fen)
    global  nn_evaluator
    if NN_TYPE == "DNN":
        nn_evaluator = DNNEvaluator.create(board, NN_TYPE, MODEL_PATH)
    else:
        nn_evaluator = NNUEEvaluator.create(board, NN_TYPE, MODEL_PATH)

    best_move = None
    best_score = 0
    best_pv = []
    pv_move = None

    last_depth_time = 0.0

    try:
        for depth in range(1, max_depth + 1):
            depth_start_time = time.perf_counter()
            check_time()

            # Check if we should stop (respects MIN_DEPTH for soft_stop)
            if should_stop_search(depth):
                break

            # After MIN_DEPTH, check if we have enough time to likely complete the next depth
            if depth > MIN_NEGAMAX_DEPTH and time_limit is not None and last_depth_time > 0:
                elapsed = time.perf_counter() - TimeControl.start_time
                remaining = time_limit - elapsed
                estimated_next_depth_time = last_depth_time * ESTIMATED_BRANCHING_FACTOR

                if remaining < estimated_next_depth_time * TIME_SAFETY_MARGIN:
                    # Not enough time to likely complete this depth
                    break

            age_heuristic_history()

            # -------- Root TT move --------
            root_key = board.zobrist_hash()
            entry = transposition_table.get(root_key)
            tt_move = entry.best_move if entry and entry.best_move in board.get_legal_moves_list() else None

            # -------- Aspiration window --------
            window = ASPIRATION_WINDOW
            retries = 0
            depth_completed = False  # Track if this depth completed successfully
            search_aborted = False  # Track if search was aborted mid-depth

            while not search_aborted:
                # First iteration should use full window
                if depth == 1:
                    alpha = -MAX_SCORE
                    beta = MAX_SCORE
                else:
                    alpha = best_score - window
                    beta = best_score + window
                alpha_orig = alpha

                current_best_score = -MAX_SCORE
                current_best_move = None
                current_best_pv = []

                for move_index, move in enumerate(ordered_moves(board, depth, pv_move, tt_move)):
                    check_time()

                    # Check for stop before searching this move
                    if TimeControl.stop_search:
                        search_aborted = True
                        break
                    if depth > MIN_NEGAMAX_DEPTH and TimeControl.soft_stop:
                        search_aborted = True
                        break

                    push_move(board, move, nn_evaluator)

                    if is_draw_by_repetition(board):
                        score = get_draw_score(board)
                        child_pv = []
                    else:
                        # Principal variation first, then late moves with PVS
                        if move_index == 0:
                            score, child_pv = negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                            score = -score
                        else:
                            score, child_pv = negamax(board, depth - 1, -alpha - 1, -alpha, allow_singular=True)
                            score = -score
                            if score > alpha:
                                score, child_pv = negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                                score = -score

                    board.pop()
                    nn_evaluator.pop()

                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move
                        current_best_pv = [move] + child_pv

                    if score > alpha:
                        alpha = score

                    if alpha >= beta:
                        break

                # If search was aborted, save partial result and exit
                if search_aborted:
                    if current_best_move is not None:
                        best_move = current_best_move
                        best_score = current_best_score
                        best_pv = current_best_pv
                    break

                # -------- SUCCESS: within aspiration window --------
                if current_best_score > alpha_orig and current_best_score < beta:
                    best_move = current_best_move
                    best_score = current_best_score
                    best_pv = current_best_pv
                    pv_move = best_move
                    depth_completed = True
                    break

                # -------- FAIL-LOW or FAIL-HIGH: widen window --------
                window *= 2
                retries += 1

                # Save partial result in case we need to stop
                if current_best_move is not None:
                    best_move = current_best_move
                    best_score = current_best_score
                    best_pv = current_best_pv

                # -------- FALLBACK: full window search --------
                if retries >= MAX_AW_RETRIES:
                    alpha = -MAX_SCORE
                    beta = MAX_SCORE
                    current_best_score = -MAX_SCORE

                    for move in ordered_moves(board, depth, pv_move, tt_move):
                        check_time()

                        # Check for stop
                        if TimeControl.stop_search:
                            search_aborted = True
                            break
                        if depth > MIN_NEGAMAX_DEPTH and TimeControl.soft_stop:
                            search_aborted = True
                            break

                        push_move(board, move, nn_evaluator)
                        score, child_pv = negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                        score = -score
                        board.pop()
                        nn_evaluator.pop()

                        if score > current_best_score:
                            current_best_score = score
                            current_best_move = move
                            current_best_pv = [move] + child_pv

                        if score > alpha:
                            alpha = score

                    # Save results from fallback search
                    if current_best_move is not None:
                        best_move = current_best_move
                        best_score = current_best_score
                        best_pv = current_best_pv
                        pv_move = best_move

                    # Only mark complete if we didn't abort
                    if not search_aborted:
                        depth_completed = True
                    break

            # Record time taken for this depth (only if completed)
            if depth_completed:
                last_depth_time = time.perf_counter() - depth_start_time

            # Break out of depth loop if search was aborted
            if search_aborted:
                break

            # Print progress with PV only if depth completed
            if depth_completed and best_pv:
                elapsed = time.perf_counter() - TimeControl.start_time
                nps = int(kpi['nodes'] / elapsed) if elapsed > 0 else 0
                print(f"info depth {depth} score cp {best_score} nodes {kpi['nodes']} nps {nps} pv {' '.join(m.uci() for m in best_pv)}", flush=True)

            # Early break to speed up testing
            if best_move is not None and expected_best_moves is not None and best_move in expected_best_moves:
                break

    except TimeoutError:
        # Return last completed depth's best move
        pass

    if best_move is None:
        board = CachedBoard(fen)
        legal = board.get_legal_moves_list()
        if legal:
            best_move = legal[0]
            best_score = evaluate_material(board)
            best_pv = [best_move]
        else:
            # No legal moves - game is over (checkmate or stalemate)
            best_move = chess.Move.null()  # Or handle differently
            best_score = evaluate_material(board)
            best_pv = []

    return best_move, best_score, best_pv


def push_move(board: CachedBoard, move, evaluator: NNEvaluator):
    """Push move on both evaluator and board using the unified interface."""
    evaluator.push_with_board(board, move)


def tokenize_san_string(san_string: str) -> list[str]:
    tokens = san_string.strip().split()
    san_moves = []
    for tok in tokens:
        if re.match(r"^\d+\.+$", tok):
            continue
        if re.match(r"^\d+\.\.\.$", tok):
            continue
        san_moves.append(tok)
    return san_moves


def pv_from_san_string(fen: str, san_string: str) -> list[chess.Move]:
    board = CachedBoard(fen)
    pv = []
    for ply, san in enumerate(tokenize_san_string(san_string)):
        move = board.parse_san(san)
        pv.append(move)
        push_move(board, move, nn_evaluator)
    return pv


def print_vars(var_names, module_name, local_scope=None):
    local_scope = local_scope or {}
    global_scope = globals()
    module_vars = {}

    if module_name:
        try:
            module = importlib.import_module(module_name)
            module_vars = vars(module)
        except ImportError:
            module_vars = {}

    for name in var_names:
        if name in local_scope:
            value = local_scope[name]
            source = "local"
        elif name in global_scope:
            value = global_scope[name]
            source = "global"
        elif name in module_vars:
            value = module_vars[name]
            source = f"module:{module_name}"
        else:
            value = "<NOT FOUND>"
            source = "missing"

        print(f"{name} = {value}")


def dump_parameters():
    print_vars([
        "IS_NUMPY_EVAL",
        "DNN_MODEL_FILEPATH",
        "IS_DNN_ENABLED",
        "QS_DEPTH_MAX_DNN_EVAL_UNCONDITIONAL",
        "QS_DEPTH_MAX_DNN_EVAL_CONDITIONAL",
        "DELTA_MAX_DNN_EVAL",
        "STAND_PAT_MAX_DNN_EVAL",
        "QS_TT_SUPPORTED",
        "DELTA_PRUNING_QS_MIN_DEPTH",
        "DELTA_PRUNING_MARGIN",
        "TACTICAL_QS_MAX_DEPTH",
        "ASPIRATION_WINDOW",
        "MAX_AW_RETRIES",
        "LMR_MOVE_THRESHOLD",
        "LMR_MIN_DEPTH",
        "NULL_MOVE_REDUCTION",
        "NULL_MOVE_MIN_DEPTH",
        "SINGULAR_MARGIN",
        "SINGULAR_EXTENSION",
        "ESTIMATED_BRANCHING_FACTOR",
        "TIME_SAFETY_MARGIN"], "engine")


def main():
    """
    Interactive loop to input FEN positions and get the best move and evaluation.
    Tracks KPIs and handles timeouts and interruptions gracefully.
    """

    while True:
        try:
            fen = input("FEN: ").strip()
            if fen.lower() == "exit":
                break
            if fen == "":
                print("Type 'exit' to quit")
                continue

            # repeated_fen = "8/8/8/p6p/P3k1pP/6p1/8/5K2 w - - 20 59"
            # board = chess.Board(repeated_fen)
            # key = chess.polyglot.zobrist_hash(board)
            # game_position_history[key] = game_position_history.get(key, 0) + 1
            # game_position_history[key] = game_position_history.get(key, 0) + 1
            # print("Eapen game_position_history.get(key, 0)=", game_position_history.get(key, 0))

            # Reset KPIs
            for key in kpi:
                kpi[key] = 0

            # Start timer
            start_time = time.perf_counter()
            move, score, pv = find_best_move(fen, max_depth=20, time_limit=30)
            end_time = time.perf_counter()

            # Record cache sizes and time
            elapsed_time = end_time - start_time
            kpi.update({
                'tt_size': len(transposition_table),
                'dec_size': len(dnn_eval_cache),
                'time': round(elapsed_time, 2),
                'nps': int(kpi['nodes'] / elapsed_time) if elapsed_time > 0 else 0
            })

            # Print KPIs
            print("\n--- Search KPIs ---")
            for key, value in kpi.items():
                print(f"{key}: {value}")

            board = CachedBoard(fen)
            print("\nBest move:", move)
            print("Evaluation:", score)
            print("PV:", pv_to_san(board, pv))
            print("PV (UCI):", " ".join(m.uci() for m in pv))
            print("-------------------\n")

        except KeyboardInterrupt:
            response = input("\nKeyboardInterrupt detected. Type 'exit' to quit, Enter to continue: ").strip()
            if response.lower() == "exit":
                break
            print("Resuming...\n")
        except Exception as e:
            print("Error:", e)
            continue


if __name__ == '__main__':
    main()