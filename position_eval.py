import chess

class PositionEval:
    def __init__(self):
        # Piece values in centipawns
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000  # Very high value to avoid capture
        }

        # Positional bonus tables (white's perspective)
        self.pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, -5, -10, 0, 0, -10, -5, 5,
            5, 10, 10, -20, -20, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

        self.knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]

        self.bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]

        self.king_table_midgame = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]

    def evaluate_position(self, board):
        """
        Evaluate the position from the perspective of the side to move.
        Returns evaluation in centipawns.
        """
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 10000 if board.turn == chess.WHITE else -10000
            elif result == "0-1":
                return -10000 if board.turn == chess.WHITE else 10000
            else:
                return 0  # Draw

        evaluation = 0

        # Material and positional evaluation
        evaluation += self._evaluate_material(board)
        evaluation += self._evaluate_positional(board)
        evaluation += self._evaluate_mobility(board)

        # Adjust perspective: if it's black's turn, negate the evaluation
        if not board.turn:
            evaluation = -evaluation

        return evaluation

    def _evaluate_material(self, board):
        """Evaluate material advantage."""
        material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value

        return material

    def _evaluate_positional(self, board):
        """Evaluate positional factors."""
        positional = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self._get_positional_value(board, piece, square)
                if piece.color == chess.WHITE:
                    positional += value
                else:
                    positional -= value

        return positional

    def _get_positional_value(self, board, piece, square):
        """Get positional bonus for a piece on a specific square."""
        # Convert to white's perspective
        if piece.color == chess.WHITE:
            index = square
        else:
            # Mirror for black pieces
            index = 63 - square

        piece_type = piece.piece_type

        if piece_type == chess.PAWN:
            return self.pawn_table[index]
        elif piece_type == chess.KNIGHT:
            return self.knight_table[index]
        elif piece_type == chess.BISHOP:
            return self.bishop_table[index]
        elif piece_type == chess.KING:
            return self.king_table_midgame[index]
        elif piece_type == chess.ROOK:
            # Rook on open/semi-open files
            return self._evaluate_rook_position(board, piece, square)

        return 0

    def _evaluate_rook_position(self, board, rook, square):
        """Evaluate rook position bonuses."""
        bonus = 0
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        # Bonus for rook on 7th rank (2nd rank for black)
        if rook.color == chess.WHITE and rank_idx == 6:
            bonus += 20
        elif rook.color == chess.BLACK and rank_idx == 1:
            bonus += 20

        # Check for open/semi-open files
        file_squares = [chess.square(file_idx, r) for r in range(8)]
        has_friendly_pawn = False
        has_enemy_pawn = False

        for sq in file_squares:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == rook.color:
                    has_friendly_pawn = True
                else:
                    has_enemy_pawn = True

        if not has_friendly_pawn:
            if not has_enemy_pawn:
                bonus += 15  # Open file
            else:
                bonus += 10  # Semi-open file

        return bonus

    def _evaluate_mobility(self, board):
        mobility = 0

        # Evaluate mobility for white
        temp_board = board.copy()
        temp_board.turn = chess.WHITE
        white_moves = len(list(temp_board.legal_moves))

        # Evaluate mobility for black
        temp_board.turn = chess.BLACK
        black_moves = len(list(temp_board.legal_moves))

        mobility = (white_moves - black_moves) * 5  # 5 centipawns per move

        return mobility


def positional_eval(board) -> int:
    if positional_eval.evaluator is None:
        positional_eval.evaluator = PositionEval()
    return positional_eval.evaluator.evaluate_position(board)
positional_eval.evaluator = None


def  main():

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break

            board = chess.Board(fen)
            print("positional score: ", positional_eval(board))

        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break


if __name__ == '__main__':
    main()

