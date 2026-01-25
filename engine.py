import importlib
import os
import random
import re
import time
import traceback
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Optional
import chess
import chess.polyglot

from cached_board import CachedBoard, move_to_int
from nn_evaluator import DNNEvaluator, NNUEEvaluator, NNEvaluator
from nn_inference import MAX_SCORE

CURR_DIR = Path(__file__).resolve().parent
HOME_DIR = "ChessDNN"

# -------- DIAGNOSTIC CONTROL --------
# Set IS_DIAGNOSTIC = True for development/debugging builds
# Or enable via UCI "debug on" command
IS_DIAGNOSTIC = False  # Master switch for diagnostic output
debug_mode = False     # Runtime toggle via UCI "debug on/off"

# Multiprocessing configuration
MAX_MP_CORES = 1  # 1 or less disables multiprocessing, UCI option "Threads"
IS_SHARED_TT_MP = False  # Whether to share TT across workers in MP mode

IS_BLAS_ENABLED = False

IS_NN_ENABLED = True
NN_TYPE = "NNUE"
# L1_QUANTIZATION defined in nn_inference.py
FULL_NN_EVAL_FREQ = 3000  # Increase to 50_000 after initial testing
DNN_MODEL_FILEPATH = CURR_DIR / f"../{HOME_DIR}" / 'model' / 'dnn.pt'
NNUE_MODEL_FILEPATH = CURR_DIR / f"../{HOME_DIR}" / 'model' / 'nnue.pt'

QS_DEPTH_MIN_NN_EVAL = 2
QS_DEPTH_MAX_NN_EVAL = 10
MAX_QS_DEPTH = 15  # REDUCED from 25 to prevent search explosion
DELTA_MAX_NN_EVAL = 50  # Score difference, below which will trigger a NN evaluation
STAND_PAT_MAX_NN_EVAL = 200

# NEW: Limit moves examined per QS ply to prevent explosion
MAX_QS_MOVES_PER_PLY = 12  # Maximum captures to examine at each QS depth
MAX_QS_MOVES_DEEP = 6  # Even fewer moves at deeper QS levels (depth > MAX_QS_DEPTH/2)

QS_TT_SUPPORTED = True
DELTA_PRUNING_QS_MIN_DEPTH = 6
DELTA_PRUNING_MARGIN = 75
TACTICAL_QS_MAX_DEPTH = 5

# NEW: Time check frequency in QS
QS_TIME_CHECK_INTERVAL = 50  # Check time every N nodes in QS (was 10 moves)

ASPIRATION_WINDOW = 50  # INCREASED from 40 to reduce retries in tactical positions
MAX_AW_RETRIES = 2  # REDUCED from 3 to fail faster to full window

LMR_MOVE_THRESHOLD = 2
LMR_MIN_DEPTH = 3  # minimum depth to apply LMR

NULL_MOVE_REDUCTION = 3  # R value (usually 2 or 3)
NULL_MOVE_MIN_DEPTH = 4

SINGULAR_MARGIN = 130  # Score difference in centipawns to trigger singular extension
SINGULAR_EXTENSION = 1  # Extra depth

# SEE Pruning - prune losing captures at low depths
SEE_PRUNING_ENABLED = False
SEE_PRUNING_MAX_DEPTH = 6  # Only apply SEE pruning at shallow depths

# Futility Pruning - skip quiet moves when position is hopeless
FUTILITY_PRUNING_ENABLED = True
FUTILITY_MARGIN = [0, 150, 300, 450]  # Margins by depth (depth 1, 2, 3)
FUTILITY_MAX_DEPTH = 3  # Only apply at depth <= 3

# Razoring - drop into quiescence when far below alpha
RAZORING_ENABLED = False
RAZORING_MARGIN = [0, 125, 250]  # Margins by depth (depth 1, 2)
RAZORING_MAX_DEPTH = 2  # Only apply at depth <= 2

# Time management
ESTIMATED_BRANCHING_FACTOR = 3.5  # Typical branching factor after pruning
TIME_SAFETY_MARGIN = 0.55  # Only start new depth if we estimate having 70%+ of needed time

MIN_NEGAMAX_DEPTH = 2  # FIXED: Allow stopping after depth 1 when time is critical
MAX_NEGAMAX_DEPTH = 20
MAX_SEARCH_TIME = 30
MAX_TABLE_SIZE = 200_000

if not IS_BLAS_ENABLED:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

MODEL_PATH = str(DNN_MODEL_FILEPATH if NN_TYPE == "DNN" else NNUE_MODEL_FILEPATH)

def is_debug_enabled() -> bool:
    """Check if diagnostic output is enabled (either via IS_DIAGNOSTIC or UCI debug on)."""
    return IS_DIAGNOSTIC or debug_mode


def set_debug_mode(enabled: bool):
    """Set debug mode from UCI command."""
    global debug_mode
    debug_mode = enabled

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
    "nn_evals": 0,
    "beta_cutoffs": 0,
    "tt_hits": 0,
    "qs_tt_hits": 0,
    "dec_hits": 0,  # DNN evaluation cache hits
    "q_depth": 0,
    "see_prunes": 0,  # SEE pruning
    "futility_prunes": 0,  # Futility pruning
    "razoring_prunes": 0,  # Razoring
}

# -------- DIAGNOSTIC COUNTERS (for debugging) --------
# These track anomalies without flooding output. Warnings printed only when threshold exceeded.
_diag = {
    "tt_illegal_moves": 0,  # TT returned illegal move
    "score_out_of_bounds": 0,  # Score exceeded MAX_SCORE
    "time_overruns": 0,  # Search ran over time limit
    "pv_illegal_moves": 0,  # PV contained illegal move
    "eval_drift": 0,  # Incremental vs full eval mismatch
    "qs_depth_exceeded": 0,  # QS hit MAX_QS_DEPTH
    "aspiration_retries": 0,  # Aspiration window retries hit max
    "best_move_none": 0,  # Search completed but best_move is None (fallback used)
    "score_instability": 0,  # Large score swings between depths
    # NEW diagnostic counters
    "qs_time_cutoff": 0,  # QS terminated early due to time
    "qs_move_limit": 0,  # QS move limit reached
    "fallback_shallow_search": 0,  # Used shallow search fallback instead of pure NN
    "aw_tactical_skip": 0,  # Skipped aspiration window due to tactical position
    "time_critical_abort": 0,  # Search aborted due to critical time pressure
}
_DIAG_WARN_THRESHOLD = 3  # Print warning after this many occurrences
_DIAG_SAMPLE_RATE = 100  # Check expensive diagnostics every N nodes
_SCORE_INSTABILITY_THRESHOLD = 200  # cp swing that triggers warning

# Track QS statistics for current search
_qs_stats = {
    "max_depth_reached": 0,
    "total_nodes": 0,
    "time_cutoffs": 0,
}


def _diag_warn(key: str, msg: str):
    """Record diagnostic event and warn if threshold exceeded (only when diagnostics enabled)."""
    _diag[key] += 1
    if not is_debug_enabled():
        return  # Skip output when diagnostics disabled
    count = _diag[key]
    # Print first occurrence, then at threshold, then every 10x threshold
    if count == 1 or count == _DIAG_WARN_THRESHOLD or (
            count > _DIAG_WARN_THRESHOLD and count % (_DIAG_WARN_THRESHOLD * 10) == 0):
        print(f"info string DIAG[{key}={count}]: {msg}", flush=True)


def diag_summary() -> str:
    """Return summary of diagnostic counters (non-zero only). Only meaningful when diagnostics enabled."""
    non_zero = {k: v for k, v in _diag.items() if v > 0}
    if non_zero:
        return "DIAG_SUMMARY: " + ", ".join(f"{k}={v}" for k, v in non_zero.items())
    return "DIAG_SUMMARY: all clear"


def diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


# Track positions seen in the current game (cleared on ucinewgame)
game_position_history: dict[int, int] = {}  # zobrist_hash -> count

# nn_evaluator: DNNEvaluator | NNUEEvaluator | None = None
if NN_TYPE == "DNN":
    nn_evaluator = DNNEvaluator.create(CachedBoard(), NN_TYPE, MODEL_PATH)  # Loads model
else:
    nn_evaluator = NNUEEvaluator.create(CachedBoard(), NN_TYPE, MODEL_PATH)

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
    # Also reset diagnostic counters for fresh tracking
    for key in _diag:
        _diag[key] = 0


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


def evaluate_nn(board: CachedBoard) -> int:
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
    kpi['nn_evals'] += 1
    score = nn_evaluator.evaluate_centipawns(board)
    dnn_eval_cache[key] = score

    # Occasionally do a full evaluation to rule out any drift errors.
    if kpi['nn_evals'] % FULL_NN_EVAL_FREQ == 0:
        full_score = nn_evaluator.evaluate_full_centipawns(board)
        if abs(full_score - score) > 10:
            _diag_warn("eval_drift", f"Incr={score} vs Full={full_score}, diff={abs(full_score - score)}")
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

    # OPTIMIZATION: Use integer key instead of tuple for faster hashing
    score += history_heuristic.get(move_to_int(move), 0)

    # Use cached gives_check
    if board.gives_check_cached(move):
        score += 50

    return score


def see(board: CachedBoard, move: chess.Move) -> int:
    """
    Simplified Static Exchange Evaluation (SEE).

    Returns the estimated material gain/loss in centipawns from the capture.
    Positive = likely winning exchange, Negative = likely losing exchange.

    This is a simplified SEE based on MVV-LVA (Most Valuable Victim - Least Valuable Attacker).
    For a full SEE, we would need to simulate the entire exchange sequence.
    """
    # Get victim and attacker types from cache
    victim_type = board.get_victim_type(move)
    attacker_type = board.get_attacker_type(move)

    if victim_type is None:
        # Not a capture - check for promotion gain
        if move.promotion:
            return PIECE_VALUES.get(move.promotion, 0) - PIECE_VALUES[chess.PAWN]
        return 0

    victim_value = PIECE_VALUES.get(victim_type, 0)
    attacker_value = PIECE_VALUES.get(attacker_type, 0)

    # Handle promotions - add the promotion piece value minus pawn value
    promotion_gain = 0
    if move.promotion:
        promotion_gain = PIECE_VALUES.get(move.promotion, 0) - PIECE_VALUES[chess.PAWN]

    # Simple MVV-LVA approximation:
    # If we capture something worth more than our piece, it's likely good.
    # If we capture something worth less with a valuable piece, it might be bad
    # unless we're protected (which we don't check in this simplified version).

    # Basic heuristic: assume the opponent can recapture with a pawn if our piece
    # lands on a square that could be defended. This is conservative.

    # Simple approximation:
    # - If victim >= attacker: good trade (we win at least the difference)
    # - If victim < attacker: potentially bad (we might lose our piece)

    if victim_value >= attacker_value:
        # Winning or equal trade
        return victim_value - attacker_value + promotion_gain
    else:
        # We're capturing with a more valuable piece
        # Pessimistic assumption: we might lose our attacker
        # But we do win the victim first
        # Net: victim - attacker (likely negative)
        return victim_value - attacker_value + promotion_gain


def see_ge(board: CachedBoard, move: chess.Move, threshold: int = 0) -> bool:
    """
    Check if SEE value of move is >= threshold.
    Used for SEE pruning - returns True if the capture is at least break-even.
    """
    return see(board, move) >= threshold


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
    Quiescence search with improved time management and move limits.

    Returns:
        Tuple of (score, pv) where pv is the list of moves in the principal variation.
    """
    global _qs_stats

    kpi['q_depth'] = max(kpi['q_depth'], q_depth)
    kpi['nodes'] += 1
    _qs_stats["total_nodes"] += 1
    _qs_stats["max_depth_reached"] = max(_qs_stats["max_depth_reached"], q_depth)

    # -------- IMPROVED: More frequent time checks in QS --------
    # Check time every QS_TIME_CHECK_INTERVAL nodes to prevent massive overruns
    if _qs_stats["total_nodes"] % QS_TIME_CHECK_INTERVAL == 0:
        check_time()

    # Hard stop always honored immediately
    if TimeControl.stop_search:
        if IS_NN_ENABLED:
            return evaluate_nn(board), []
        else:
            return evaluate_material(board), []

    # -------- NEW: Soft stop honored in deep QS to prevent time explosions --------
    if TimeControl.soft_stop and q_depth > MAX_QS_DEPTH // 2:
        _qs_stats["time_cutoffs"] += 1
        _diag_warn("qs_time_cutoff", f"QS soft-stopped at depth {q_depth}")
        if IS_NN_ENABLED:
            return evaluate_nn(board), []
        else:
            return evaluate_material(board), []

    # -------- Hard depth limit to prevent search explosion --------
    if q_depth > MAX_QS_DEPTH:
        # DIAG: Track QS depth limit hits
        _diag_warn("qs_depth_exceeded", f"QS hit depth {q_depth}, fen={board.fen()[:40]}")
        # Return static evaluation when QS goes too deep
        if IS_NN_ENABLED:
            return evaluate_nn(board), []
        else:
            return evaluate_material(board), []

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

    # Stand-pat is not valid when in check. It is likely that there will wil a move that is better
    # than stand-pat
    if not is_check:
        is_dnn_eval = False
        if IS_NN_ENABLED and q_depth <= QS_DEPTH_MIN_NN_EVAL:
            stand_pat = evaluate_nn(board)
            is_dnn_eval = True
        else:
            stand_pat = evaluate_material(board)

        if (not is_dnn_eval and IS_NN_ENABLED and q_depth <= QS_DEPTH_MAX_NN_EVAL
                and abs(stand_pat) < STAND_PAT_MAX_NN_EVAL
                and abs(stand_pat - beta) < DELTA_MAX_NN_EVAL):
            stand_pat = evaluate_nn(board)
            is_dnn_eval = True

        if stand_pat >= beta:
            kpi['beta_cutoffs'] += 1
            return stand_pat, []  # ✅ Return stand_pat, not beta

        # Big delta pruning - can't possibly reach alpha
        if stand_pat + PIECE_VALUES[chess.QUEEN] < alpha:
            return stand_pat, []  # ✅ Return stand_pat (it's the best we can do)

        if (not is_dnn_eval and IS_NN_ENABLED
                and q_depth <= QS_DEPTH_MAX_NN_EVAL
                and abs(stand_pat) < STAND_PAT_MAX_NN_EVAL
                and (stand_pat > alpha or abs(stand_pat - alpha) < DELTA_MAX_NN_EVAL)):
            stand_pat = evaluate_nn(board)

        best_score = stand_pat  # ✅ Initialize best_score with stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    moves_searched = 0

    # -------- NEW: Determine move limit based on depth --------
    move_limit = MAX_QS_MOVES_DEEP if q_depth > MAX_QS_DEPTH // 2 else MAX_QS_MOVES_PER_PLY

    # Pre-compute move info for this position
    board.precompute_move_info()

    for move in ordered_moves_q_search(board):
        # -------- NEW: Enforce move limit to prevent explosion --------
        if not is_check and moves_searched >= move_limit:
            _diag_warn("qs_move_limit", f"QS move limit ({move_limit}) at depth {q_depth}")
            break

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

        should_update_nn = q_depth <= QS_DEPTH_MAX_NN_EVAL
        if should_update_nn:
            push_move(board, move, nn_evaluator)
        else:
            board.push(move)  # Only push board, not evaluator

        score, child_pv = quiescence(board, -beta, -alpha, q_depth + 1)
        score = -score
        board.pop()

        if should_update_nn:
            nn_evaluator.pop()

        moves_searched += 1

        # -------- IMPROVED: Check time more frequently --------
        if moves_searched % 5 == 0:
            check_time()
            if TimeControl.stop_search:
                break  # Exit loop gracefully, return best_score below
            # Also check soft_stop in deep QS
            if TimeControl.soft_stop and q_depth > MAX_QS_DEPTH // 2:
                _qs_stats["time_cutoffs"] += 1
                break

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
    # Only check hard stop here - soft_stop is handled at root level
    # This ensures current depth completes before stopping
    if TimeControl.stop_search:
        # Return material eval to keep push/pop balanced
        return evaluate_material(board), []

    # -------- Draw detection --------
    if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
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
        # DIAG: Verify TT move is legal (corruption detection)
        if tt_move and tt_move not in board.get_legal_moves_list():
            _diag_warn("tt_illegal_moves", f"TT move {tt_move.uci()} illegal in {board.fen()[:50]}")
            tt_move = None  # Ignore corrupted TT move
    else:
        tt_move = None

    # -------- Quiescence if depth == 0 --------
    if depth == 0:
        return quiescence(board, alpha, beta, 1)

    in_check = board.is_check()

    # -------- Razoring (drop into quiescence when far below alpha) --------
    # At low depths, if we're way below alpha, just go to quiescence
    if (RAZORING_ENABLED
            and not in_check
            and depth <= RAZORING_MAX_DEPTH
            and depth >= 1):
        # Get static eval (use material for speed)
        static_eval = evaluate_material(board)
        margin = RAZORING_MARGIN[depth] if depth < len(RAZORING_MARGIN) else RAZORING_MARGIN[-1]

        if static_eval + margin <= alpha:
            # Position is so bad that even with a margin, we're below alpha
            # Drop into quiescence to see if tactics can save us
            qs_score, qs_pv = quiescence(board, alpha, beta, 1)
            if qs_score <= alpha:
                kpi['razoring_prunes'] += 1
                return qs_score, qs_pv

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

    # -------- Futility Pruning Setup --------
    # Compute static eval once for futility pruning (only needed at low depths)
    futility_pruning_applicable = False
    static_eval = None
    if (FUTILITY_PRUNING_ENABLED
            and not in_check
            and depth <= FUTILITY_MAX_DEPTH
            and depth >= 1
            and abs(alpha) < MAX_SCORE - 100):  # Not near mate scores
        static_eval = evaluate_material(board)
        futility_margin = FUTILITY_MARGIN[depth] if depth < len(FUTILITY_MARGIN) else FUTILITY_MARGIN[-1]
        # If static eval + margin is still below alpha, futility pruning may apply
        if static_eval + futility_margin <= alpha:
            futility_pruning_applicable = True

    for move_index, move in enumerate(moves):
        # Use cached move info
        is_capture = board.is_capture_cached(move)
        gives_check = board.gives_check_cached(move)

        # -------- SEE Pruning (prune losing captures at low depths) --------
        if (SEE_PRUNING_ENABLED
                and depth <= SEE_PRUNING_MAX_DEPTH
                and is_capture
                and not in_check
                and move != tt_move
                and move_index > 0):  # Don't prune the first move
            # Prune captures with negative SEE at low depths
            # Threshold increases with depth (more aggressive pruning at lower depths)
            see_threshold = -20 * depth  # e.g., -20 at depth 1, -40 at depth 2, etc.
            if not see_ge(board, move, see_threshold):
                kpi['see_prunes'] += 1
                continue  # Skip this losing capture

        # -------- Futility Pruning (skip quiet moves when position is hopeless) --------
        if (futility_pruning_applicable
                and not is_capture
                and not gives_check
                and move != tt_move
                and move_index > 0):  # Don't prune the first move or TT move
            # Check if the move is not a killer move
            is_killer = False
            if depth is not None and 0 <= depth < len(killer_moves):
                is_killer = (move == killer_moves[depth][0] or move == killer_moves[depth][1])

            if not is_killer:
                # This quiet move is unlikely to raise the score above alpha
                kpi['futility_prunes'] += 1
                continue

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
                # OPTIMIZATION: Use integer key for faster hashing
                key_hist = move_to_int(move)
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
            # DIAG: Track when TT contains illegal PV move
            _diag_warn("pv_illegal_moves", f"PV move {move.uci()} illegal at depth {len(pv)}")
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
    """
    Control dictionary size by removing oldest entries.
    OPTIMIZED: Instead of popping one at a time (O(n) for each pop),
    we rebuild the dict with only the most recent entries.
    """
    current_size = len(table)
    if current_size > max_dict_size:
        # Calculate how many entries to keep
        entries_to_keep = max_dict_size * 3 // 4  # Keep 75%

        # For a regular dict (Python 3.7+), iteration order is insertion order
        # So we want to keep the LAST entries_to_keep items
        entries_to_remove = current_size - entries_to_keep

        # Much faster: get keys to remove, then delete them
        # Using islice to get first N keys (oldest entries)
        from itertools import islice
        keys_to_remove = list(islice(table.keys(), entries_to_remove))

        for key in keys_to_remove:
            del table[key]


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
        temp_board.push(move)  # Only push to board, don't modify nn_evaluator

    return " ".join(san_moves)


def find_best_move(fen, max_depth=MAX_NEGAMAX_DEPTH, time_limit=None, clear_tt=True, expected_best_moves=None) -> \
        Tuple[Optional[chess.Move], int, List[chess.Move], int, float]:
    """
    Finds the best move for a given FEN using iterative deepening negamax with alpha-beta pruning,
    aspiration windows, TT, quiescence, null-move pruning, LMR, singular extensions, and heuristics.

    Args:
        fen: FEN string of the position
        max_depth: Maximum search depth
        time_limit: Time limit in seconds (None for no limit)
        expected_best_moves: For testing - stop early if best move matches
        clear_tt: If True, clear transposition tables before search. Set to False
                  on ponderhit to benefit from warm TT.

    Returns:
        Tuple of (best_move, score, pv, nodes, nps)
    """
    global _qs_stats

    # -------- Initialize time control --------
    TimeControl.time_limit = time_limit
    TimeControl.stop_search = False
    TimeControl.soft_stop = False  # Reset soft stop
    TimeControl.start_time = time.perf_counter()
    nodes_start = kpi['nodes']
    nps = 0

    # -------- NEW: Reset QS statistics for this search --------
    _qs_stats = {
        "max_depth_reached": 0,
        "total_nodes": 0,
        "time_cutoffs": 0,
    }

    diag_print(f"TimeControl: stop={TimeControl.stop_search}, soft={TimeControl.soft_stop}, time_limit={time_limit}")

    # -------- Clear search tables & heuristics --------
    # -------- DIAGNOSTIC: Track initialization steps --------
    init_start = time.perf_counter()

    for i in range(len(killer_moves)):
        killer_moves[i] = [None, None]
    history_heuristic.clear()

    # Only clear TT if requested (preserve on ponderhit for warm cache)
    if clear_tt:
        transposition_table.clear()
        qs_transposition_table.clear()

    # -------- CRITICAL: NN cache trimming can be slow if cache is huge --------
    cache_size_before = len(dnn_eval_cache)
    cache_trim_start = time.perf_counter()
    control_dict_size(dnn_eval_cache, MAX_TABLE_SIZE)
    cache_trim_time = time.perf_counter() - cache_trim_start
    cache_size_after = len(dnn_eval_cache)

    if cache_trim_time > 0.1:
        diag_print(f"DEBUG: NN cache trim took {cache_trim_time:.3f}s "
                   f"({cache_size_before} -> {cache_size_after} entries)")

    tables_cleared_time = time.perf_counter() - init_start
    if tables_cleared_time > 0.1:
        diag_print(f"DEBUG: Tables cleared in {tables_cleared_time:.3f}s")

    # -------- Initialize board and evaluator --------
    board_init_start = time.perf_counter()
    board = CachedBoard(fen)
    board_init_time = time.perf_counter() - board_init_start
    if board_init_time > 0.1:
        diag_print(f"DEBUG: Board init took {board_init_time:.3f}s")

    # -------- CRITICAL: Check for stop before NN reset (which can be slow) --------
    if TimeControl.stop_search:
        diag_print(f"DEBUG: Stop received before NN reset, aborting")
        # Return first legal move as fallback
        legal = board.get_legal_moves_list()
        if legal:
            return legal[0], 0, [legal[0]], 0, 0
        return chess.Move.null(), 0, [], 0, 0

    nn_reset_start = time.perf_counter()
    nn_evaluator.reset(board)
    nn_reset_time = time.perf_counter() - nn_reset_start
    if nn_reset_time > 0.1:
        diag_print(f"DEBUG: NN reset took {nn_reset_time:.3f}s")

    total_init_time = time.perf_counter() - init_start
    if total_init_time > 0.2:
        diag_print(f"DEBUG: Total init took {total_init_time:.3f}s")

    # Start with no result - fallback computed lazily only if needed
    best_move = None
    best_score = 0
    best_pv = []
    pv_move = None
    prev_depth_score = None  # For score stability tracking

    last_depth_time = 0.0

    # -------- NEW: Track if position appears tactical (large score swings) --------
    is_tactical_position = False

    for depth in range(1, max_depth + 1):
        depth_start_time = time.perf_counter()
        check_time()

        # -------- NEW: Debug output for search progress --------
        if depth <= 3:
            diag_print(f"DEBUG: Starting depth {depth}, stop={TimeControl.stop_search}, soft={TimeControl.soft_stop}")

        # Check if we should stop (respects MIN_DEPTH for soft_stop)
        if should_stop_search(depth):
            diag_print(f"DEBUG: Stopping at depth {depth} due to should_stop_search")
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

        # -------- NEW: Skip aspiration window for tactical positions or when score is extreme --------
        use_full_window = (depth == 1 or is_tactical_position or
                           (prev_depth_score is not None and abs(prev_depth_score) > 500))
        if use_full_window and depth > 1:
            _diag_warn("aw_tactical_skip",
                       f"Skipping AW at depth {depth}, tactical={is_tactical_position}, score={prev_depth_score}")

        while not search_aborted:
            # First iteration or tactical positions use full window
            if use_full_window:
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

                # -------- NEW: Debug output for move progress at depth 1 --------
                if depth == 1 and move_index < 3:
                    diag_print(f"DEBUG: depth 1 move {move_index}: {move.uci()}, stop={TimeControl.stop_search}")

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

                # Check if search was aborted during negamax
                if TimeControl.stop_search or (depth > MIN_NEGAMAX_DEPTH and TimeControl.soft_stop):
                    search_aborted = True
                    # Still save this move's result if it's our only one
                    if current_best_move is None:
                        current_best_move = move
                        current_best_score = score
                        current_best_pv = [move] + child_pv
                    break

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                    current_best_pv = [move] + child_pv

                if score > alpha:
                    alpha = score

                if alpha >= beta:
                    break

            # If search was aborted, only save partial result if we have NO completed result
            # FIXED: Don't overwrite a completed depth's result with an incomplete one
            if search_aborted:
                if best_move is None and current_best_move is not None:
                    best_move = current_best_move
                    best_score = current_best_score
                    best_pv = current_best_pv
                break

            # -------- SUCCESS: within aspiration window (or using full window) --------
            if use_full_window or (current_best_score > alpha_orig and current_best_score < beta):
                best_move = current_best_move
                best_score = current_best_score
                best_pv = current_best_pv
                pv_move = best_move
                depth_completed = True
                break

            # -------- FAIL-LOW or FAIL-HIGH: widen window --------
            window *= 2
            retries += 1

            # DIAG: Track excessive aspiration retries
            if retries >= MAX_AW_RETRIES:
                _diag_warn("aspiration_retries", f"depth={depth} hit MAX_AW_RETRIES, score={current_best_score}")
                # -------- NEW: Mark position as tactical for future depths --------
                is_tactical_position = True

            # Save partial result ONLY if we have NO completed result yet
            # FIXED: Don't overwrite a completed depth's result with an incomplete aspiration retry
            if best_move is None and current_best_move is not None:
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

                    # Check if search was aborted during negamax
                    if TimeControl.stop_search or (depth > MIN_NEGAMAX_DEPTH and TimeControl.soft_stop):
                        search_aborted = True
                        if current_best_move is None:
                            current_best_move = move
                            current_best_score = score
                            current_best_pv = [move] + child_pv
                        break

                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move
                        current_best_pv = [move] + child_pv

                    if score > alpha:
                        alpha = score

                # Save results from fallback search ONLY if completed (not aborted)
                # Or if we have no result yet (best_move is None)
                if not search_aborted:
                    if current_best_move is not None:
                        best_move = current_best_move
                        best_score = current_best_score
                        best_pv = current_best_pv
                        pv_move = best_move
                    depth_completed = True
                elif best_move is None and current_best_move is not None:
                    # Aborted but no previous result - save partial
                    best_move = current_best_move
                    best_score = current_best_score
                    best_pv = current_best_pv
                break

        # Record time taken for this depth (only if completed)
        if depth_completed:
            last_depth_time = time.perf_counter() - depth_start_time

            # DIAG: Check for score instability (large swings between depths)
            if prev_depth_score is not None and depth > 2:
                score_diff = abs(best_score - prev_depth_score)
                if score_diff > _SCORE_INSTABILITY_THRESHOLD:
                    _diag_warn("score_instability",
                               f"depth {depth}: {prev_depth_score} -> {best_score} (diff={score_diff})")
                    # -------- NEW: Mark as tactical if large swing --------
                    is_tactical_position = True
            prev_depth_score = best_score

        # Break out of depth loop if search was aborted
        if search_aborted:
            break

        # Print progress with PV only if depth completed
        if depth_completed and best_pv:
            elapsed = time.perf_counter() - TimeControl.start_time
            nps = int((kpi['nodes'] - nodes_start) / elapsed) if elapsed > 0 else 0
            print(
                f"info depth {depth} score cp {best_score} nodes {kpi['nodes']} nps {nps} pv {' '.join(m.uci() for m in best_pv)}",
                flush=True)

        # Early break to speed up testing
        if best_move is not None and expected_best_moves is not None and best_move in expected_best_moves:
            break

    # -------- NEW: Print QS statistics if significant --------
    if _qs_stats["max_depth_reached"] > MAX_QS_DEPTH // 2 or _qs_stats["time_cutoffs"] > 0:
        diag_print(f"QS_STATS: max_depth={_qs_stats['max_depth_reached']}, "
                   f"nodes={_qs_stats['total_nodes']}, time_cutoffs={_qs_stats['time_cutoffs']}")

    # -------- IMPROVED fallback: shallow tactical search instead of pure NN eval --------
    if best_move is None:
        # DIAG: Track when fallback is needed
        _diag_warn("best_move_none", f"No depth completed, using shallow search fallback, fen={fen[:40]}")
        diag_print(f"Computing shallow search fallback (no depth completed)...")

        # -------- NEW: Debug output to understand why no depth completed --------
        elapsed = time.perf_counter() - TimeControl.start_time if TimeControl.start_time else 0
        diag_print(f"DEBUG fallback: elapsed={elapsed:.2f}s, stop={TimeControl.stop_search}, "
                   f"soft={TimeControl.soft_stop}, time_limit={time_limit}")

        # Reset to clean state for fallback evaluation
        board = CachedBoard(fen)
        nn_evaluator.reset(board)
        legal = board.get_legal_moves_list()

        if legal:
            best_move = legal[0]  # Default to first legal move
            best_score = -MAX_SCORE

            # -------- NEW: Do a shallow 1-ply search with captures/checks --------
            # This provides some tactical awareness without calling quiescence
            _diag_warn("fallback_shallow_search", f"Shallow search with {len(legal)} moves")

            fallback_aborted = False
            for move in legal:
                # -------- CRITICAL: Check for stop signal --------
                if TimeControl.stop_search:
                    diag_print(f"Fallback aborted by stop signal")
                    fallback_aborted = True
                    break

                nn_evaluator.push_with_board(board, move)

                # Check for immediate game-ending conditions
                if board.is_checkmate():
                    # We just delivered checkmate!
                    score = MAX_SCORE - board.ply()
                    board.pop()
                    nn_evaluator.pop()
                    best_move = move
                    best_score = score
                    best_pv = [move]
                    diag_print(f"Fallback found checkmate: {move.uci()}")
                    break

                if board.is_stalemate() or board.is_insufficient_material():
                    score = 0  # Draw
                else:
                    # Do a 1-ply tactical check: look at opponent's best response
                    # Only examine captures and checks to limit explosion
                    opp_best = MAX_SCORE  # From opponent's perspective (we want to minimize this)
                    opp_moves_checked = 0
                    max_opp_moves = 8  # Limit opponent moves to check

                    board.precompute_move_info()
                    for opp_move in board.get_legal_moves_list():
                        # -------- CRITICAL: Check for stop signal in inner loop --------
                        if TimeControl.stop_search:
                            break

                        # Only consider captures, checks, or promotions
                        is_tactical = (board.is_capture_cached(opp_move) or
                                       board.gives_check_cached(opp_move) or
                                       opp_move.promotion is not None)

                        if not is_tactical and opp_moves_checked > 0:
                            continue

                        # -------- FIXED: Use push_move helper which handles both board and nn_evaluator --------
                        push_move(board, opp_move, nn_evaluator)

                        # Check for mate threats
                        if board.is_checkmate():
                            opp_score = MAX_SCORE - board.ply()  # Opponent wins
                        else:
                            # Use NN eval for the resulting position (already pushed)
                            opp_score = -evaluate_nn(board)  # Negate because it's opponent's view

                        board.pop()
                        nn_evaluator.pop()
                        opp_best = min(opp_best, opp_score)
                        opp_moves_checked += 1

                        if opp_moves_checked >= max_opp_moves:
                            break

                    # If no tactical responses found, use static eval
                    if opp_moves_checked == 0:
                        score = -evaluate_nn(board)
                    else:
                        score = -opp_best  # Our score is negative of opponent's best

                board.pop()
                nn_evaluator.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

            best_pv = [best_move]
            if not fallback_aborted:
                diag_print(f"Shallow fallback: {len(legal)} moves, best={best_move.uci()} score={best_score}cp")
        else:
            # No legal moves - game is over (checkmate or stalemate)
            best_move = chess.Move.null()
            best_score = evaluate_material(board)
            best_pv = []

    # DIAG: Check for time overrun (only when time limit was set)
    if time_limit is not None:
        elapsed = time.perf_counter() - TimeControl.start_time
        overrun = elapsed - time_limit
        if overrun > 0.5:  # More than 500ms over
            _diag_warn("time_overruns",
                       f"Search overran by {overrun:.2f}s (limit={time_limit:.2f}s, actual={elapsed:.2f}s)")

    # -------- NEW: Additional diagnostic for severe overruns --------
    if time_limit is not None:
        elapsed = time.perf_counter() - TimeControl.start_time
        if elapsed > time_limit * 3:  # More than 3x the time limit
            _diag_warn("time_critical_abort",
                       f"SEVERE overrun: {elapsed:.2f}s vs {time_limit:.2f}s limit, QS_max_depth={_qs_stats['max_depth_reached']}")

    # DIAG: Check score bounds
    if abs(best_score) > MAX_SCORE:
        _diag_warn("score_out_of_bounds", f"Score {best_score} exceeds MAX_SCORE {MAX_SCORE}")

    return best_move, best_score, best_pv, kpi['nodes'] - nodes_start, nps


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
        board.push(move)  # Only push to board, don't modify nn_evaluator
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
    print_vars(["L1_QUANTIZATION"], "nn_inference")
    print_vars([
        "MAX_MP_CORES",
        "IS_SHARED_TT_MP",
        "IS_NN_ENABLED",
        "NN_TYPE",
        "MODEL_PATH",
        "IS_BLAS_ENABLED",
        "QS_DEPTH_MIN_NN_EVAL",
        "QS_DEPTH_MAX_NN_EVAL",
        "MAX_QS_DEPTH",
        "DELTA_MAX_NN_EVAL",
        "STAND_PAT_MAX_NN_EVAL",
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
        "SEE_PRUNING_ENABLED",
        "SEE_PRUNING_MAX_DEPTH",
        "FUTILITY_PRUNING_ENABLED",
        "FUTILITY_MARGIN",
        "FUTILITY_MAX_DEPTH",
        "RAZORING_ENABLED",
        "RAZORING_MARGIN",
        "RAZORING_MAX_DEPTH",
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

            # Reset KPIs
            for key in kpi:
                kpi[key] = 0

            # Start timer
            start_time = time.perf_counter()
            move, score, pv, _, _ = find_best_move(fen, max_depth=20, time_limit=5)
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
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()