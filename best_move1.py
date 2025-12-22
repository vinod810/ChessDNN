import chess
import sys

# Piece values for evaluation (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Simple Piece-Square Tables (PST) to encourage better positioning
# These are from the perspective of White. We mirror them for Black.
pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]

knighttable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

bishoptable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

rooktable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
]

queentable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

kingtable = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
]


class ChessAI:
    def __init__(self, depth=3):
        self.depth = depth
        self.nodes_explored = 0

    def evaluate_board(self, board):
        """
        Static evaluation function.
        Returns a score in centipawns. Positive is good for White, negative for Black.
        """
        if board.is_checkmate():
            if board.turn:
                return -99999  # Black wins (White is to move and mated)
            else:
                return 99999  # White wins

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        material_score = 0

        # Calculate material and positional scores
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            value = PIECE_VALUES[piece.piece_type]

            # Positional score from PST
            # Note: chess.SQUARES goes 0-63 (A1 to H8).
            # Our tables are defined 0-63 top-left to bottom-right or bottom-up depending on convention.
            # python-chess squares are 0=a1, 1=b1 ... 63=h8.
            # To map our tables (which look like the board visually):
            # We need to mirror rank for White (index ^ 56) to match the visual array layout above if they are Top-Down.
            # Actually, standard PSTs usually are defined Rank 8 to Rank 1.
            # Let's assume the arrays above are Rank 1 (bottom) to Rank 8 (top) for simplicity in this implementation,
            # or we just map carefully.

            # Let's map simpler: index is 0..63.
            # For White: a1 is index 0.
            # The tables above: Let's assume index 0 is A1.
            pst_value = 0
            if piece.piece_type == chess.PAWN:
                pst_value = pawntable[square]
            elif piece.piece_type == chess.KNIGHT:
                pst_value = knighttable[square]
            elif piece.piece_type == chess.BISHOP:
                pst_value = bishoptable[square]
            elif piece.piece_type == chess.ROOK:
                pst_value = rooktable[square]
            elif piece.piece_type == chess.QUEEN:
                pst_value = queentable[square]
            elif piece.piece_type == chess.KING:
                pst_value = kingtable[square]

            if piece.color == chess.WHITE:
                material_score += value + pst_value
            else:
                # Mirror square for Black positional play
                # Mirroring rank: square ^ 56
                mirror_square = square ^ 56

                pst_value = 0
                if piece.piece_type == chess.PAWN:
                    pst_value = pawntable[mirror_square]
                elif piece.piece_type == chess.KNIGHT:
                    pst_value = knighttable[mirror_square]
                elif piece.piece_type == chess.BISHOP:
                    pst_value = bishoptable[mirror_square]
                elif piece.piece_type == chess.ROOK:
                    pst_value = rooktable[mirror_square]
                elif piece.piece_type == chess.QUEEN:
                    pst_value = queentable[mirror_square]
                elif piece.piece_type == chess.KING:
                    pst_value = kingtable[mirror_square]

                material_score -= (value + pst_value)

        return material_score

    def quiescence_search(self, board, alpha, beta):
        """
        Quiescence Search:
        Extends the search at leaf nodes to process all capture moves.
        This prevents the 'horizon effect' where the engine stops calculating
        in the middle of a capture sequence.
        """
        self.nodes_explored += 1
        stand_pat = self.evaluate_board(board)

        # Fail-hard beta cutoff
        if stand_pat >= beta:
            return beta

        # Delta pruning could go here (if stand_pat + BIG_PIECE < alpha, return alpha)

        if alpha < stand_pat:
            alpha = stand_pat

        # Generate only capture moves
        capture_moves = [move for move in board.legal_moves if board.is_capture(move)]

        # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        # This helps Alpha-Beta prune faster inside Quiescence
        capture_moves.sort(key=lambda m: self.score_move(board, m), reverse=True)

        for move in capture_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board, depth, alpha, beta):
        """
        Implements Alpha-Beta Pruning using the NegaMax variant.

        NegaMax simplifies the Minimax algorithm by relying on the fact that
        max(a, b) == -min(-a, -b). This allows us to use a single function
        for both White and Black turns, rather than separate 'maximize' and
        'minimize' blocks.
        """
        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)

        self.nodes_explored += 1

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            if board.is_check():
                return -99999 + (self.depth - depth)  # Checkmate, prefer shorter mates
            else:
                return 0  # Stalemate

        # Order moves
        legal_moves.sort(key=lambda m: self.score_move(board, m), reverse=True)

        for move in legal_moves:
            board.push(move)
            # -negamax returns score for us (the parent)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta  # Snip
            if score > alpha:
                alpha = score

        return alpha

    def score_move(self, board, move):
        """
        Heuristic to order moves:
        1. Captures (MVV-LVA)
        2. Promotions
        3. History/Killers (not implemented here, simple version)
        """
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if victim:
                score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[
                    board.piece_at(move.from_square).piece_type]

        if move.promotion:
            score += 900

        return score

    def get_best_move(self, fen):
        board = chess.Board(fen)
        print(f"Analyzing Position: {fen}")
        print(f"Side to move: {'White' if board.turn else 'Black'}")

        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda m: self.score_move(board, m), reverse=True)

        # Iterative Deepening could go here, but we'll stick to fixed depth
        print(f"Searching to depth {self.depth}...")

        for move in legal_moves:
            board.push(move)
            # We want to minimize the opponent's response
            board_value = -self.negamax(board, self.depth - 1, -beta, -alpha)
            board.pop()

            print(f"Move: {move} Score: {board_value}")

            if board_value > best_value:
                best_value = board_value
                best_move = move

            # Update root alpha
            alpha = max(alpha, board_value)

        return best_move, best_value


# --- Usage Example ---

if __name__ == "__main__":
    # Example FEN: Position where White has a tactical win or advantage
    # Or start pos: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Tactical puzzle FEN (White to move):
    tactical_fen = "r1bqk2r/pppp1ppp/2n5/2b1p3/2B1P1n1/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 1 6"

    ai = ChessAI(depth=1)
    best_move, score = ai.get_best_move(tactical_fen)

    print("-" * 30)
    print(f"Best Move found: {best_move}")
    print(f"Evaluation Score: {score}")
    print(f"Nodes Explored: {ai.nodes_explored}")