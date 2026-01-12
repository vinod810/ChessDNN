from abc import ABC, abstractmethod


class NNEvaluator(ABC):
    """
    Abstract evaluator for neural network position evaluation.
    Handles both NNUE and DNN with unified interface.
    """

    @abstractmethod
    def push(self, move: chess.Move):
        """Make a move and update internal state."""
        pass

    @abstractmethod
    def pop(self):
        """Undo last move and restore state."""
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate current position (returns raw NN output)."""
        pass

    def evaluate_centipawns(self) -> int:
        """Evaluate and convert to centipawns (common for both)."""
        return int(self.evaluate() * 400)

    @abstractmethod
    def get_board(self) -> chess.Board:
        """Get current board state."""
        pass

    @staticmethod
    def create(board: chess.Board, nn_type: str, model_path: str):
        """Factory method to create appropriate evaluator."""
        if nn_type.upper() == "DNN":
            return DNNEvaluator(board, model_path)
        elif nn_type.upper() == "NNUE":
            return NNUEEvaluator(board, model_path)
        else:
            raise ValueError(f"Unknown NN type: {nn_type}")


class DNNEvaluator(NNEvaluator):
    """DNN-based evaluator with incremental updates."""

    def __init__(self, board: chess.Board, model_path: str):
        from nn_train import DNNInference, DNNIncrementalUpdater

        self.inference = DNNInference(model_path)
        self.updater = DNNIncrementalUpdater(board)

        # Initialize accumulators
        white_feat, black_feat = self.updater.get_features_both()
        self.inference._refresh_accumulator(white_feat, True)
        self.inference._refresh_accumulator(black_feat, False)

    def push(self, move: chess.Move):
        old_white = set(self.updater.white_features)
        old_black = set(self.updater.black_features)

        self.updater.push(move)

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
        change_record = self.updater.pop()

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

    def evaluate(self) -> float:
        if self.updater.board.is_game_over():
            if self.updater.board.is_checkmate():
                return -INF + self.updater.board.ply()
            return 0.0

        features = self.updater.get_features()
        perspective = self.updater.board.turn == chess.WHITE
        return self.inference.evaluate_incremental(features, perspective)

    def get_board(self) -> chess.Board:
        return self.updater.board


class NNUEEvaluator(NNEvaluator):
    """NNUE-based evaluator with incremental updates."""

    def __init__(self, board: chess.Board, model_path: str):
        from nn_train import NNUEInference, IncrementalFeatureUpdater

        self.inference = NNUEInference(model_path)
        self.updater = IncrementalFeatureUpdater(board)

        # Initialize accumulator
        white_feat, black_feat = self.updater.get_features_unsorted()
        self.inference._refresh_accumulator(white_feat, black_feat)

    def push(self, move: chess.Move):
        old_white = set(self.updater.white_features)
        old_black = set(self.updater.black_features)

        self.updater.push(move)

        new_white = set(self.updater.white_features)
        new_black = set(self.updater.black_features)

        # NNUE updates both accumulators in one call
        self.inference.update_accumulator(
            new_white - old_white,
            old_white - new_white,
            new_black - old_black,
            old_black - new_black
        )

    def pop(self):
        change_record = self.updater.pop()

        # NNUE update_accumulator signature
        self.inference.update_accumulator(
            change_record['white_removed'],
            change_record['white_added'],
            change_record['black_removed'],
            change_record['black_added']
        )

    def evaluate(self) -> float:
        if self.updater.board.is_game_over():
            if self.updater.board.is_checkmate():
                return -INF + self.updater.board.ply()
            return 0.0

        white_feat, black_feat = self.updater.get_features_unsorted()
        stm = self.updater.board.turn == chess.WHITE
        return self.inference.evaluate_incremental(white_feat, black_feat, stm)

    def get_board(self) -> chess.Board:
        return self.updater.board

"""
# engine.py

nn_type = "NNUE"  # Config at top of file
MODEL_PATH = "model/nnue.pt"

class ChessEngine:
    def __init__(self):
        self.evaluator = None
        self.tt = {}
        
    def search(self, board: chess.Board, depth: int):
        # Create evaluator for this search
        self.evaluator = NNEvaluator.create(board, nn_type, MODEL_PATH)
        
        return self.negamax(depth, -INF, INF)
    
    def negamax(self, depth, alpha, beta):
        ...
        if depth == 0:
            return self.evaluator.evaluate_centipawns(), []
        
        for move in moves:
            self.evaluator.push(move)
            score, pv = self.negamax(depth-1, -beta, -alpha)
            self.evaluator.pop()
            ..
"""