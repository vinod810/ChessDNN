"""
Enhanced CachedBoard - Efficient chess board with intelligent caching.

This module provides a chess.Board subclass with caching for expensive computations,
incremental Zobrist hash updates, and piece-square table evaluation.
"""

import chess
import chess.polyglot
from typing import Optional, List, Iterator, Dict
from dataclasses import dataclass
from copy import deepcopy


# ========== Piece Values and PST Tables ==========

PIECE_VALUES: Dict[chess.PieceType, int] = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# fmt: off
PST_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
PST_ROOK = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PST_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
PST_KING_MIDDLEGAME = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
PST_KING_ENDGAME = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]
# fmt: on

_PST_TABLES = {
    chess.PAWN: PST_PAWN, chess.KNIGHT: PST_KNIGHT, chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK, chess.QUEEN: PST_QUEEN, chess.KING: PST_KING_MIDDLEGAME,
}

def get_pst_value(piece_type, square, color, is_endgame=False):
    if piece_type == chess.KING:
        table = PST_KING_ENDGAME if is_endgame else PST_KING_MIDDLEGAME
    else:
        table = _PST_TABLES.get(piece_type)
        if table is None:
            return 0
    if color == chess.BLACK:
        square = chess.square_mirror(square)
    return table[square]

# Zobrist constants
_ZOBRIST_CASTLING_BASE = 768
_ZOBRIST_EP_BASE = 772
_ZOBRIST_TURN = 780

def _zobrist_piece_key(piece_type, color, square):
    piece_offset = piece_type - 1
    if color == chess.BLACK:
        piece_offset += 6
    return chess.polyglot.POLYGLOT_RANDOM_ARRAY[64 * piece_offset + square]


@dataclass(slots=True)
class CacheState:
    zobrist_hash: Optional[int] = None
    legal_moves: Optional[List[chess.Move]] = None
    is_any_capture_available: Optional[bool] = None
    has_non_pawn_material: Optional[Dict[chess.Color, bool]] = None
    is_check: Optional[bool] = None
    is_checkmate: Optional[bool] = None
    is_stalemate: Optional[bool] = None
    is_game_over: Optional[bool] = None
    piece_count: Optional[int] = None
    is_insufficient_material: Optional[bool] = None
    can_claim_draw: Optional[bool] = None
    material_evaluation: Optional[int] = None
    is_endgame: Optional[bool] = None
    gives_check_cache: Optional[Dict[chess.Move, bool]] = None

    def clear(self):
        self.zobrist_hash = None
        self.legal_moves = None
        self.is_any_capture_available = None
        self.has_non_pawn_material = None
        self.is_check = None
        self.is_checkmate = None
        self.is_stalemate = None
        self.is_game_over = None
        self.piece_count = None
        self.is_insufficient_material = None
        self.can_claim_draw = None
        self.material_evaluation = None
        self.is_endgame = None
        self.gives_check_cache = None


@dataclass(slots=True)
class MoveInfo:
    move: chess.Move
    captured_piece: Optional[chess.Piece] = None
    was_en_passant: bool = False
    was_castling: bool = False
    previous_castling_rights: chess.Bitboard = 0
    previous_ep_square: Optional[chess.Square] = None


class CachedBoard(chess.Board):
    def __init__(self, fen: Optional[str] = chess.STARTING_FEN):
        self._cache_stack: List[CacheState] = [CacheState()]
        self._move_info_stack: List[MoveInfo] = []
        self._use_incremental_zobrist: bool = True
        self._use_incremental_material: bool = True
        super().__init__(fen)
        self._cache_stack = [CacheState()]
        self._move_info_stack = []

    @property
    def _cache(self) -> CacheState:
        return self._cache_stack[-1]

    def push(self, move: chess.Move) -> None:
        move_info = MoveInfo(
            move=move,
            captured_piece=self._get_captured_piece(move),
            was_en_passant=self.is_en_passant(move),
            was_castling=self.is_castling(move),
            previous_castling_rights=self.castling_rights,
            previous_ep_square=self.ep_square,
        )
        super().push(move)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(CacheState())

    def _get_captured_piece(self, move: chess.Move) -> Optional[chess.Piece]:
        if self.is_en_passant(move):
            return chess.Piece(chess.PAWN, not self.turn)
        return self.piece_at(move.to_square)

    def pop(self) -> chess.Move:
        move = super().pop()
        if len(self._cache_stack) > 1:
            self._cache_stack.pop()
        else:
            self._cache_stack[-1].clear()
        if self._move_info_stack:
            self._move_info_stack.pop()
        return move

    def push_uci(self, uci: str) -> chess.Move:
        move = chess.Move.from_uci(uci)
        self.push(move)
        return move

    def push_san(self, san: str) -> chess.Move:
        move = self.parse_san(san)
        self.push(move)
        return move

    def reset(self) -> None:
        super().reset()
        self._cache_stack = [CacheState()]
        self._move_info_stack = []

    def set_fen(self, fen: str) -> None:
        super().set_fen(fen)
        self._cache_stack = [CacheState()]
        self._move_info_stack = []

    def set_board_fen(self, fen: str) -> None:
        super().set_board_fen(fen)
        self._invalidate_current_cache()

    def set_piece_at(self, square, piece, promoted=False) -> None:
        super().set_piece_at(square, piece, promoted)
        self._invalidate_current_cache()

    def remove_piece_at(self, square):
        piece = super().remove_piece_at(square)
        self._invalidate_current_cache()
        return piece

    def clear_board(self) -> None:
        super().clear_board()
        self._invalidate_current_cache()

    def set_castling_fen(self, castling_fen: str) -> None:
        super().set_castling_fen(castling_fen)
        self._invalidate_current_cache()

    def _invalidate_current_cache(self) -> None:
        self._cache.clear()
        self._move_info_stack.clear()

    def clear_cache(self) -> None:
        self._cache.clear()

    def copy(self, stack: bool = True) -> "CachedBoard":
        board = CachedBoard(None)
        board.set_fen(self.fen())
        if stack and len(self.move_stack) > 0:
            moves = list(self.move_stack)
            board.reset()
            board.set_fen(chess.STARTING_FEN if not self.chess960 else self.starting_fen)
            for move in moves:
                board.push(move)
            board._cache_stack[-1] = deepcopy(self._cache)
        else:
            board._cache_stack = [deepcopy(self._cache)]
            board._move_info_stack = []
        return board

    def __copy__(self):
        return self.copy(stack=False)

    def __deepcopy__(self, memo):
        return self.copy(stack=True)

    def zobrist_hash(self, verify: bool = False) -> int:
        if self._cache.zobrist_hash is None:
            computed_incrementally = False
            if (self._use_incremental_zobrist and len(self._cache_stack) > 1
                and len(self._move_info_stack) > 0):
                parent_cache = self._cache_stack[-2]
                if parent_cache.zobrist_hash is not None:
                    move_info = self._move_info_stack[-1]
                    self._cache.zobrist_hash = self._compute_incremental_zobrist(
                        parent_cache.zobrist_hash, move_info)
                    computed_incrementally = True
            if self._cache.zobrist_hash is None:
                self._cache.zobrist_hash = chess.polyglot.zobrist_hash(self)
            if verify and computed_incrementally:
                actual = chess.polyglot.zobrist_hash(self)
                if self._cache.zobrist_hash != actual:
                    self._cache.zobrist_hash = actual
        return self._cache.zobrist_hash

    def _compute_incremental_zobrist(self, parent_hash: int, move_info: MoveInfo) -> int:
        move = move_info.move
        h = parent_hash
        piece = self.piece_at(move.to_square)
        if piece is None:
            return chess.polyglot.zobrist_hash(self)
        moving_color = piece.color
        original_type = chess.PAWN if move.promotion else piece.piece_type

        h ^= _zobrist_piece_key(original_type, moving_color, move.from_square)
        h ^= _zobrist_piece_key(piece.piece_type, moving_color, move.to_square)

        if move_info.captured_piece is not None:
            captured = move_info.captured_piece
            if move_info.was_en_passant:
                ep_sq = chess.square(chess.square_file(move.to_square),
                                     chess.square_rank(move.from_square))
                h ^= _zobrist_piece_key(captured.piece_type, captured.color, ep_sq)
            else:
                h ^= _zobrist_piece_key(captured.piece_type, captured.color, move.to_square)

        if move_info.was_castling:
            if chess.square_file(move.to_square) == 6:
                rook_from = chess.square(7, chess.square_rank(move.from_square))
                rook_to = chess.square(5, chess.square_rank(move.from_square))
            else:
                rook_from = chess.square(0, chess.square_rank(move.from_square))
                rook_to = chess.square(3, chess.square_rank(move.from_square))
            h ^= _zobrist_piece_key(chess.ROOK, moving_color, rook_from)
            h ^= _zobrist_piece_key(chess.ROOK, moving_color, rook_to)

        old_rights, new_rights = move_info.previous_castling_rights, self.castling_rights
        if bool(old_rights & chess.BB_H1) != bool(new_rights & chess.BB_H1):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE]
        if bool(old_rights & chess.BB_A1) != bool(new_rights & chess.BB_A1):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE + 1]
        if bool(old_rights & chess.BB_H8) != bool(new_rights & chess.BB_H8):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE + 2]
        if bool(old_rights & chess.BB_A8) != bool(new_rights & chess.BB_A8):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE + 3]

        old_ep, new_ep = move_info.previous_ep_square, self.ep_square
        if old_ep is not None:
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_EP_BASE + chess.square_file(old_ep)]
        if new_ep is not None:
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_EP_BASE + chess.square_file(new_ep)]

        h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_TURN]
        return h

    def get_legal_moves_list(self) -> List[chess.Move]:
        if self._cache.legal_moves is None:
            self._cache.legal_moves = list(super().legal_moves)
        return self._cache.legal_moves

    def legal_moves_iter(self) -> Iterator[chess.Move]:
        return iter(self.get_legal_moves_list())

    def count_legal_moves(self) -> int:
        return len(self.get_legal_moves_list())

    def has_legal_moves(self) -> bool:
        if self._cache.legal_moves is not None:
            return len(self._cache.legal_moves) > 0
        return any(True for _ in super().legal_moves)

    def is_any_capture_available(self) -> bool:
        if self._cache.is_any_capture_available is None:
            self._cache.is_any_capture_available = any(
                self.is_capture(m) for m in self.get_legal_moves_list())
        return self._cache.is_any_capture_available

    is_any_move_capture = is_any_capture_available

    def get_capture_moves(self) -> List[chess.Move]:
        return [m for m in self.get_legal_moves_list() if self.is_capture(m)]

    def gives_check(self, move: chess.Move) -> bool:
        if self._cache.gives_check_cache is None:
            self._cache.gives_check_cache = {}
        if move not in self._cache.gives_check_cache:
            self._cache.gives_check_cache[move] = super().gives_check(move)
        return self._cache.gives_check_cache[move]

    def has_non_pawn_material(self, color: Optional[chess.Color] = None) -> bool:
        if color is None:
            color = self.turn
        if self._cache.has_non_pawn_material is None:
            self._cache.has_non_pawn_material = {}
        if color not in self._cache.has_non_pawn_material:
            result = any(self.pieces(pt, color)
                        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
            self._cache.has_non_pawn_material[color] = result
        return self._cache.has_non_pawn_material[color]

    def is_check(self) -> bool:
        if self._cache.is_check is None:
            self._cache.is_check = super().is_check()
        return self._cache.is_check

    def is_checkmate(self) -> bool:
        if self._cache.is_checkmate is None:
            self._cache.is_checkmate = super().is_checkmate()
        return self._cache.is_checkmate

    def is_stalemate(self) -> bool:
        if self._cache.is_stalemate is None:
            self._cache.is_stalemate = super().is_stalemate()
        return self._cache.is_stalemate

    def is_game_over(self) -> bool:
        if self._cache.is_game_over is None:
            self._cache.is_game_over = super().is_game_over()
        return self._cache.is_game_over

    def is_insufficient_material(self) -> bool:
        if self._cache.is_insufficient_material is None:
            self._cache.is_insufficient_material = super().is_insufficient_material()
        return self._cache.is_insufficient_material

    def can_claim_draw(self) -> bool:
        if self._cache.can_claim_draw is None:
            self._cache.can_claim_draw = super().can_claim_draw()
        return self._cache.can_claim_draw

    def piece_count(self) -> int:
        if self._cache.piece_count is None:
            self._cache.piece_count = chess.popcount(self.occupied)
        return self._cache.piece_count

    def is_endgame_position(self) -> bool:
        if self._cache.is_endgame is None:
            wq = chess.popcount(self.pieces_mask(chess.QUEEN, chess.WHITE))
            bq = chess.popcount(self.pieces_mask(chess.QUEEN, chess.BLACK))
            if wq == 0 and bq == 0:
                self._cache.is_endgame = True
            else:
                self._cache.is_endgame = True
                for color in (chess.WHITE, chess.BLACK):
                    if chess.popcount(self.pieces_mask(chess.QUEEN, color)) > 0:
                        minors = chess.popcount(
                            self.pieces_mask(chess.KNIGHT, color) |
                            self.pieces_mask(chess.BISHOP, color))
                        rooks = chess.popcount(self.pieces_mask(chess.ROOK, color))
                        if rooks > 0 or minors > 1:
                            self._cache.is_endgame = False
                            break
        return self._cache.is_endgame

    def material_evaluation(self, verify: bool = False) -> int:
        if self._cache.material_evaluation is None:
            computed_incrementally = False
            if (self._use_incremental_material and len(self._cache_stack) > 1
                and len(self._move_info_stack) > 0):
                parent_cache = self._cache_stack[-2]
                if parent_cache.material_evaluation is not None:
                    move_info = self._move_info_stack[-1]
                    self._cache.material_evaluation = self._compute_incremental_material(
                        parent_cache.material_evaluation, move_info,
                        self._cache_stack[-2].is_endgame)
                    computed_incrementally = True
            if self._cache.material_evaluation is None:
                self._cache.material_evaluation = self._compute_material_evaluation()
            if verify and computed_incrementally:
                actual = self._compute_material_evaluation()
                if self._cache.material_evaluation != actual:
                    self._cache.material_evaluation = actual
        return self._cache.material_evaluation

    def _compute_material_evaluation(self) -> int:
        our_mat, their_mat = 0, 0
        our_color = self.turn
        is_eg = self.is_endgame_position()
        for sq in chess.SQUARES:
            piece = self.piece_at(sq)
            if piece is None:
                continue
            val = PIECE_VALUES[piece.piece_type] + get_pst_value(piece.piece_type, sq, piece.color, is_eg)
            if piece.color == our_color:
                our_mat += val
            else:
                their_mat += val
        return our_mat - their_mat

    def _compute_incremental_material(self, parent_eval, move_info, parent_is_endgame):
        move = move_info.move
        is_eg = self.is_endgame_position()
        if parent_is_endgame is not None and parent_is_endgame != is_eg:
            return self._compute_material_evaluation()

        new_eval = -parent_eval
        piece = self.piece_at(move.to_square)
        if piece is None:
            return self._compute_material_evaluation()

        moving_color = piece.color
        original_type = chess.PAWN if move.promotion else piece.piece_type

        old_pst = get_pst_value(original_type, move.from_square, moving_color, is_eg)
        new_pst = get_pst_value(piece.piece_type, move.to_square, moving_color, is_eg)
        new_eval += old_pst - new_pst

        if move.promotion:
            new_eval -= PIECE_VALUES[piece.piece_type] - PIECE_VALUES[chess.PAWN]

        if move_info.captured_piece is not None:
            cap = move_info.captured_piece
            if move_info.was_en_passant:
                ep_sq = chess.square(chess.square_file(move.to_square),
                                     chess.square_rank(move.from_square))
                cap_pst = get_pst_value(cap.piece_type, ep_sq, not moving_color, is_eg)
            else:
                cap_pst = get_pst_value(cap.piece_type, move.to_square, not moving_color, is_eg)
            new_eval -= PIECE_VALUES[cap.piece_type] + cap_pst

        return new_eval

    def get_cache_depth(self) -> int:
        return len(self._cache_stack)

    def get_cache_info(self) -> Dict[str, bool]:
        c = self._cache
        return {
            "zobrist_hash": c.zobrist_hash is not None,
            "legal_moves": c.legal_moves is not None,
            "is_check": c.is_check is not None,
            "material_evaluation": c.material_evaluation is not None,
        }


def is_any_move_capture(board: chess.Board) -> bool:
    """Standalone compatibility function."""
    if isinstance(board, CachedBoard):
        return board.is_any_capture_available()
    return any(board.is_capture(m) for m in board.legal_moves)


if __name__ == "__main__":
    board = CachedBoard()
    print(f"Initial hash: {board.zobrist_hash()}")
    board.push_san("e4")
    print(f"After e4: {board.zobrist_hash()}")
    board.push_san("e5")
    print(f"After e5: {board.zobrist_hash()}")
    print(f"Material eval: {board.material_evaluation()}")
    print(f"Legal moves: {len(board.get_legal_moves_list())}")
    print(f"Is check: {board.is_check()}")
    print(f"Has non-pawn material: {board.has_non_pawn_material()}")