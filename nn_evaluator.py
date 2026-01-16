from abc import ABC, abstractmethod
from typing import Tuple

import chess
from nn_inference import NNUEIncrementalUpdater
from nn_inference import DNNIncrementalUpdater
from nn_inference import load_model, MAX_SCORE


class NNEvaluator(ABC):
    """
    Abstract evaluator for neural network position evaluation.
    Handles both NNUE and DNN with unified interface.

    IMPORTANT: This class does NOT manage board state. The caller (e.g., engine.py)
    is responsible for calling board.push() and board.pop(). This class only
    maintains the incremental evaluation state (accumulators, feature trackers).

    Supports two evaluation modes:
    - Incremental: Efficient for search (push/pop updates accumulators)
    - Full: Standalone evaluation without incremental state

    Usage pattern:
        evaluator.push(board, move)  # Update evaluator state
        board.push(move)             # Caller updates board
        score = evaluator.evaluate(board)
        board.pop()                  # Caller restores board
        evaluator.pop()              # Restore evaluator state
    """

    @abstractmethod
    def push(self, board_before_push: chess.Board, move: chess.Move):
        """
        Update internal state for a move. Does NOT modify the board.

        Args:
            board_before_push: Board state BEFORE the move
            move: Move being made

        Note: Caller must call board.push(move) separately after this.
        """
        pass

    @abstractmethod
    def pop(self):
        """
        Restore internal state to before the last push. Does NOT modify the board.

        Note: Caller must call board.pop() separately (typically before this).
        """
        pass

    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate current position using incremental evaluation.
        Requires proper push/pop state management.

        Args:
            board: Current board state (must match internal state)

        Returns:
            Raw NN output (approximately in [-1, 1])
        """
        pass

    @abstractmethod
    def evaluate_full(self, board: chess.Board) -> float:
        """
        Evaluate position using full matrix multiplication.
        Does not use or affect incremental state.

        Useful for:
        - One-off evaluations
        - Debugging/validation against incremental results
        - Positions not reachable via push/pop from initial position

        Returns:
            Raw NN output (approximately in [-1, 1])
        """
        pass

    def evaluate_centipawns(self, board: chess.Board) -> int:
        """Evaluate using incremental method and convert to centipawns."""
        return int(self.evaluate(board) * 400)

    def evaluate_full_centipawns(self, board: chess.Board) -> int:
        """Evaluate using full method and convert to centipawns."""
        return int(self.evaluate_full(board) * 400)

    def validate_incremental(self, board: chess.Board, tolerance: float = 1e-5) -> bool:
        """
        Validate that incremental and full evaluation match.
        Useful for debugging.

        Returns:
            True if evaluations match within tolerance
        """
        inc_eval = self.evaluate(board)
        full_eval = self.evaluate_full(board)
        return abs(inc_eval - full_eval) < tolerance

    @staticmethod
    def create(board: chess.Board, nn_type: str, model_path: str) -> 'NNEvaluator':
        """Factory method to create appropriate evaluator."""
        if nn_type.upper() == "DNN":
            return DNNEvaluator(board, model_path)
        elif nn_type.upper() == "NNUE":
            return NNUEEvaluator(board, model_path)
        else:
            raise ValueError(f"Unknown NN type: {nn_type}")

    @staticmethod
    def evaluate_position(board: chess.Board, nn_type: str, model_path: str) -> float:
        """
        Convenience method for one-off position evaluation.
        Does not create incremental state - just evaluates and returns.

        Args:
            board: Position to evaluate
            nn_type: "NNUE" or "DNN"
            model_path: Path to model file

        Returns:
            Raw NN output (approximately in [-1, 1])
        """
        inference = load_model(model_path, nn_type.upper())
        return inference.evaluate_board(board)

    @staticmethod
    def evaluate_position_centipawns(board: chess.Board, nn_type: str, model_path: str) -> int:
        """
        Convenience method for one-off position evaluation in centipawns.
        """
        return int(NNEvaluator.evaluate_position(board, nn_type, model_path) * 400)


class DNNEvaluator(NNEvaluator):
    """DNN-based evaluator with incremental updates."""

    def __init__(self, board: chess.Board, model_path: str):
        self.inference = load_model(model_path, "DNN")
        self.updater = DNNIncrementalUpdater(board)

        # Initialize accumulators for both perspectives
        white_feat, black_feat = self.updater.get_features_both()
        self.inference.refresh_accumulator(white_feat, True)
        self.inference.refresh_accumulator(black_feat, False)

    def push(self, board_before_push: chess.Board, move: chess.Move):
        """Update internal state for a move. Does NOT modify the board."""
        old_white = set(self.updater.white_features)
        old_black = set(self.updater.black_features)

        # Update feature tracker
        self.updater.push(board_before_push, move)

        new_white = set(self.updater.white_features)
        new_black = set(self.updater.black_features)

        # Update both accumulators
        self.inference.update_accumulator(
            new_white - old_white, old_white - new_white, True
        )
        self.inference.update_accumulator(
            new_black - old_black, old_black - new_black, False
        )

    def pop(self):
        """Restore internal state to before the last push. Does NOT modify the board."""
        change_record = self.updater.pop()

        # Reverse the accumulator changes
        self.inference.update_accumulator(
            change_record['white_removed'],
            change_record['white_added'],
            True
        )
        self.inference.update_accumulator(
            change_record['black_removed'],
            change_record['black_added'],
            False
        )

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate using incremental accumulators."""
        if board.is_game_over():
            if board.is_checkmate():
                return -MAX_SCORE + board.ply()
            return 0.0

        perspective = board.turn == chess.WHITE
        return self.inference.evaluate_incremental(perspective)

    def evaluate_full(self, board: chess.Board) -> float:
        """Evaluate using full matrix multiplication (no incremental state)."""
        if board.is_game_over():
            if board.is_checkmate():
                return -MAX_SCORE + board.ply()
            return 0.0

        return self.inference.evaluate_board(board)

    def reset(self, board: chess.Board):
        """
        Reset incremental state to match a new board position.
        Use when jumping to a position not reachable via push/pop.
        """
        self.updater = DNNIncrementalUpdater(board)
        white_feat, black_feat = self.updater.get_features_both()
        self.inference.refresh_accumulator(white_feat, True)
        self.inference.refresh_accumulator(black_feat, False)


class NNUEEvaluator(NNEvaluator):
    """NNUE-based evaluator with incremental updates."""

    def __init__(self, board: chess.Board, model_path: str):
        self.inference = load_model(model_path, "NNUE")
        self.updater = NNUEIncrementalUpdater(board)

        # Initialize accumulators
        white_feat, black_feat = self.updater.get_features_unsorted()
        self.inference.refresh_accumulator(white_feat, black_feat)

    def push(self, board_before_push: chess.Board, move: chess.Move):
        """
        Not supported for NNUE. Use update_pre_push() and update_post_push() instead.
        """
        raise NotImplementedError(
            "NNUEEvaluator.push() is not supported. "
            "Use update_pre_push() before board.push() and "
            "update_post_push() after board.push() instead."
        )

    def update_pre_push(self, board_before_push: chess.Board, move: chess.Move) -> Tuple:
        """
        Phase 1 of two-phase push. Call BEFORE board.push(move).

        Returns:
            Tuple of (is_white_king_move, is_black_king_move, change_record)
            Pass these to update_post_push() after calling board.push(move).
        """
        return self.updater.update_pre_push(board_before_push, move)

    def update_post_push(self, board_after_push: chess.Board,
                         is_white_king_move: bool,
                         is_black_king_move: bool,
                         change_record: dict):
        """
        Phase 2 of two-phase push. Call AFTER board.push(move).

        Args:
            board_after_push: Board state after the move was pushed
            is_white_king_move: From update_pre_push return value
            is_black_king_move: From update_pre_push return value
            change_record: From update_pre_push return value
        """
        self.updater.update_post_push(board_after_push, is_white_king_move,
                                      is_black_king_move, change_record)

        # Update accumulators
        if is_white_king_move or is_black_king_move:
            white_feat, black_feat = self.updater.get_features_unsorted()
            self.inference.refresh_accumulator(white_feat, black_feat)
        else:
            self.inference.update_accumulator(
                change_record['white_added'],
                change_record['white_removed'],
                change_record['black_added'],
                change_record['black_removed']
            )

    def pop(self):
        """Restore internal state to before the last push. Does NOT modify the board."""
        change_record = self.updater.pop()

        # Restore accumulators
        if change_record['white_king_moved'] or change_record['black_king_moved']:
            # King moves require full refresh
            white_feat, black_feat = self.updater.get_features_unsorted()
            self.inference.refresh_accumulator(white_feat, black_feat)
        else:
            # Regular move: reverse incremental updates
            self.inference.update_accumulator(
                change_record['white_removed'],
                change_record['white_added'],
                change_record['black_removed'],
                change_record['black_added']
            )

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate using incremental accumulators."""
        if board.is_game_over():
            if board.is_checkmate():
                return -MAX_SCORE + board.ply()
            return 0.0

        stm = board.turn == chess.WHITE
        return self.inference.evaluate_incremental(stm)

    def evaluate_full(self, board: chess.Board) -> float:
        """Evaluate using full matrix multiplication (no incremental state)."""
        if board.is_game_over():
            if board.is_checkmate():
                return -MAX_SCORE + board.ply()
            return 0.0

        return self.inference.evaluate_board(board)

    def reset(self, board: chess.Board):
        """
        Reset incremental state to match a new board position.
        Use when jumping to a position not reachable via push/pop.
        """
        self.updater = NNUEIncrementalUpdater(board)
        white_feat, black_feat = self.updater.get_features_unsorted()
        self.inference.refresh_accumulator(white_feat, black_feat)


# =============================================================================
# Example Usage
# =============================================================================
"""
# Basic usage with incremental evaluation (board managed by caller):

board = chess.Board()
evaluator = NNEvaluator.create(board, "NNUE", "model/nnue.pt")

# Evaluate initial position
score = evaluator.evaluate_centipawns(board)
print(f"Initial: {score} cp")

# Make moves - evaluator and board updated separately
move = chess.Move.from_uci("e2e4")
evaluator.push(board, move)  # Update evaluator state (board unchanged)
board.push(move)             # Update board state
score = evaluator.evaluate_centipawns(board)
print(f"After e4: {score} cp")

# Undo moves - board first, then evaluator
board.pop()
evaluator.pop()
score = evaluator.evaluate_centipawns(board)
print(f"After undo: {score} cp")


# Two-phase push for NNUE (most efficient when caller manages board):

evaluator = NNUEEvaluator(board, "model/nnue.pt")
move = chess.Move.from_uci("e2e4")

# Phase 1: Before board.push()
pre_push_data = evaluator.update_pre_push(board, move)

# Caller pushes the board
board.push(move)

# Phase 2: After board.push()
evaluator.update_post_push(board, *pre_push_data)

# Now evaluate
score = evaluator.evaluate_centipawns(board)


# Full evaluation (standalone, no incremental state needed):

board = chess.Board("r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
score = NNEvaluator.evaluate_position_centipawns(board, "NNUE", "model/nnue.pt")
print(f"Position score: {score} cp")

# Or using an existing evaluator (doesn't affect incremental state):
score = evaluator.evaluate_full_centipawns(board)


# In a search function (engine.py pattern):

class ChessEngine:
    def __init__(self, nn_type: str = "NNUE", model_path: str = "model/nnue.pt"):
        self.nn_type = nn_type
        self.model_path = model_path
        self.evaluator = None

    def search(self, board: chess.Board, depth: int):
        # Create evaluator for this search
        self.evaluator = NNEvaluator.create(board, self.nn_type, self.model_path)
        return self._negamax(board, depth, -float('inf'), float('inf'))

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float):
        if depth == 0 or board.is_game_over():
            return self.evaluator.evaluate_centipawns(board), []

        best_score = -float('inf')
        best_pv = []

        for move in board.legal_moves:
            # Update evaluator state, then board
            self.evaluator.push(board, move)
            board.push(move)

            score, pv = self._negamax(board, depth - 1, -beta, -alpha)
            score = -score

            # Restore board, then evaluator state
            board.pop()
            self.evaluator.pop()

            if score > best_score:
                best_score = score
                best_pv = [move] + pv

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score, best_pv
"""