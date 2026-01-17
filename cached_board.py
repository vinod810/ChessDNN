"""
CachedBoard - Efficient chess board wrapper with intelligent caching.

This module provides a chess.Board wrapper with caching for expensive computations,
incremental Zobrist hash updates, and piece-square table evaluation.

Uses composition instead of inheritance to prepare for C++ library replacement.
"""

import chess
import chess.polyglot
from typing import Optional, List, Dict
from dataclasses import dataclass

# ========== Constants ==========

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Zobrist constants
_ZOBRIST_CASTLING_BASE = 768
_ZOBRIST_EP_BASE = 772
_ZOBRIST_TURN = 780

# ========== PST Tables ==========
# fmt: off
_PST_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_PST_ROOK = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
_PST_KING_MG = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
_PST_KING_EG = [
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
    chess.PAWN: _PST_PAWN,
    chess.KNIGHT: _PST_KNIGHT,
    chess.BISHOP: _PST_BISHOP,
    chess.ROOK: _PST_ROOK,
    chess.QUEEN: _PST_QUEEN,
    chess.KING: _PST_KING_MG,
}


def _get_pst_value(piece_type: int, square: int, color: bool, is_endgame: bool = False) -> int:
    """Get piece-square table value for a piece."""
    if piece_type == chess.KING:
        table = _PST_KING_EG if is_endgame else _PST_KING_MG
    else:
        table = _PST_TABLES.get(piece_type)
        if table is None:
            return 0
    if color == chess.BLACK:
        square = chess.square_mirror(square)
    return table[square]


def _zobrist_piece_key(piece_type: int, color: bool, square: int) -> int:
    """Get Zobrist key for a piece at a square."""
    piece_offset = piece_type - 1
    if color == chess.BLACK:
        piece_offset += 6
    return chess.polyglot.POLYGLOT_RANDOM_ARRAY[64 * piece_offset + square]


# ========== Data Classes ==========

@dataclass(slots=True)
class _CacheState:
    """Cached computations for a board position."""
    zobrist_hash: Optional[int] = None
    legal_moves: Optional[List[chess.Move]] = None
    has_non_pawn_material: Optional[Dict[bool, bool]] = None
    is_check: Optional[bool] = None
    is_checkmate: Optional[bool] = None
    is_game_over: Optional[bool] = None
    material_evaluation: Optional[int] = None
    is_endgame: Optional[bool] = None
    # Pre-computed move info
    move_is_capture: Optional[Dict[chess.Move, bool]] = None
    move_gives_check: Optional[Dict[chess.Move, bool]] = None
    move_victim_type: Optional[Dict[chess.Move, Optional[int]]] = None
    move_attacker_type: Optional[Dict[chess.Move, Optional[int]]] = None


@dataclass(slots=True)
class _MoveInfo:
    """Information about a move for incremental updates."""
    move: chess.Move
    captured_piece: Optional[chess.Piece] = None
    was_en_passant: bool = False
    was_castling: bool = False
    previous_castling_rights: int = 0
    previous_ep_square: Optional[int] = None


# ========== CachedBoard ==========

class CachedBoard:
    """
    Chess board wrapper with caching for expensive computations.

    Uses composition to wrap chess.Board, making it easier to swap
    the underlying implementation with a C++ library later.
    """

    __slots__ = ('_board', '_cache_stack', '_move_info_stack')

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN):
        """Initialize board from FEN string. Pass None for empty board."""
        self._cache_stack: List[_CacheState] = [_CacheState()]
        self._move_info_stack: List[_MoveInfo] = []
        self._board = chess.Board(fen) if fen else chess.Board(None)

    # ==================== Properties (delegate to inner board) ====================

    @property
    def _cache(self) -> _CacheState:
        return self._cache_stack[-1]

    @property
    def turn(self) -> bool:
        return self._board.turn

    @property
    def fullmove_number(self) -> int:
        return self._board.fullmove_number

    @property
    def occupied(self) -> int:
        return self._board.occupied

    @property
    def castling_rights(self) -> int:
        return self._board.castling_rights

    @property
    def ep_square(self) -> Optional[int]:
        return self._board.ep_square

    @property
    def move_stack(self) -> List[chess.Move]:
        return self._board.move_stack

    @property
    def legal_moves(self):
        """Iterator over legal moves (for compatibility with chess.Board)."""
        return self._board.legal_moves

    # ==================== Core Methods ====================

    def push(self, move: chess.Move) -> None:
        """Make a move on the board."""
        # Capture move info before pushing
        move_info = _MoveInfo(
            move=move,
            captured_piece=self._get_captured_piece(move),
            was_en_passant=self._board.is_en_passant(move),
            was_castling=self._board.is_castling(move),
            previous_castling_rights=self._board.castling_rights,
            previous_ep_square=self._board.ep_square,
        )
        self._board.push(move)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(_CacheState())

    def pop(self) -> chess.Move:
        """Unmake the last move."""
        move = self._board.pop()
        if len(self._cache_stack) > 1:
            self._cache_stack.pop()
        if self._move_info_stack:
            self._move_info_stack.pop()
        return move

    def _get_captured_piece(self, move: chess.Move) -> Optional[chess.Piece]:
        """Get the piece captured by a move (before the move is made)."""
        if self._board.is_en_passant(move):
            return chess.Piece(chess.PAWN, not self._board.turn)
        return self._board.piece_at(move.to_square)

    def copy(self, stack: bool = True) -> "CachedBoard":
        """Create a copy of the board."""
        board = CachedBoard(None)
        if stack and self._board.move_stack:
            # Replay all moves to build up state
            board._board.set_fen(chess.STARTING_FEN)
            for move in self._board.move_stack:
                board.push(move)
        else:
            board._board.set_fen(self._board.fen())
        return board

    def set_fen(self, fen: str) -> None:
        """Set position from FEN string."""
        self._board.set_fen(fen)
        self._cache_stack = [_CacheState()]
        self._move_info_stack = []

    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()

    # ==================== Delegated Methods ====================

    def piece_at(self, square: int) -> Optional[chess.Piece]:
        return self._board.piece_at(square)

    def king(self, color: bool) -> Optional[int]:
        return self._board.king(color)

    def san(self, move: chess.Move) -> str:
        return self._board.san(move)

    def parse_san(self, san: str) -> chess.Move:
        return self._board.parse_san(san)

    def is_en_passant(self, move: chess.Move) -> bool:
        return self._board.is_en_passant(move)

    def is_castling(self, move: chess.Move) -> bool:
        return self._board.is_castling(move)

    def is_capture(self, move: chess.Move) -> bool:
        return self._board.is_capture(move)

    def is_repetition(self, count: int = 3) -> bool:
        return self._board.is_repetition(count)

    def can_claim_fifty_moves(self) -> bool:
        return self._board.can_claim_fifty_moves()

    def ply(self) -> int:
        return self._board.ply()

    def pieces(self, piece_type: int, color: bool) -> chess.SquareSet:
        return self._board.pieces(piece_type, color)

    def pieces_mask(self, piece_type: int, color: bool) -> int:
        return self._board.pieces_mask(piece_type, color)

    # ==================== Cached Methods ====================

    def zobrist_hash(self) -> int:
        """Get Zobrist hash with incremental update optimization."""
        if self._cache.zobrist_hash is None:
            # Try incremental computation
            if len(self._cache_stack) > 1 and self._move_info_stack:
                parent_hash = self._cache_stack[-2].zobrist_hash
                if parent_hash is not None:
                    self._cache.zobrist_hash = self._compute_incremental_zobrist(
                        parent_hash, self._move_info_stack[-1])
            # Fallback to full computation
            if self._cache.zobrist_hash is None:
                self._cache.zobrist_hash = chess.polyglot.zobrist_hash(self._board)
        return self._cache.zobrist_hash

    def _compute_incremental_zobrist(self, parent_hash: int, move_info: _MoveInfo) -> int:
        """Compute Zobrist hash incrementally from parent position."""
        move = move_info.move
        h = parent_hash

        piece = self._board.piece_at(move.to_square)
        if piece is None:
            return chess.polyglot.zobrist_hash(self._board)

        moving_color = piece.color
        original_type = chess.PAWN if move.promotion else piece.piece_type

        # XOR out old piece position, XOR in new
        h ^= _zobrist_piece_key(original_type, moving_color, move.from_square)
        h ^= _zobrist_piece_key(piece.piece_type, moving_color, move.to_square)

        # Handle captures
        if move_info.captured_piece is not None:
            captured = move_info.captured_piece
            if move_info.was_en_passant:
                ep_sq = chess.square(chess.square_file(move.to_square),
                                     chess.square_rank(move.from_square))
                h ^= _zobrist_piece_key(captured.piece_type, captured.color, ep_sq)
            else:
                h ^= _zobrist_piece_key(captured.piece_type, captured.color, move.to_square)

        # Handle castling (rook movement)
        if move_info.was_castling:
            rank = chess.square_rank(move.from_square)
            if chess.square_file(move.to_square) == 6:  # Kingside
                h ^= _zobrist_piece_key(chess.ROOK, moving_color, chess.square(7, rank))
                h ^= _zobrist_piece_key(chess.ROOK, moving_color, chess.square(5, rank))
            else:  # Queenside
                h ^= _zobrist_piece_key(chess.ROOK, moving_color, chess.square(0, rank))
                h ^= _zobrist_piece_key(chess.ROOK, moving_color, chess.square(3, rank))

        # Handle castling rights changes
        old_rights, new_rights = move_info.previous_castling_rights, self._board.castling_rights
        if bool(old_rights & chess.BB_H1) != bool(new_rights & chess.BB_H1):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE]
        if bool(old_rights & chess.BB_A1) != bool(new_rights & chess.BB_A1):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE + 1]
        if bool(old_rights & chess.BB_H8) != bool(new_rights & chess.BB_H8):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE + 2]
        if bool(old_rights & chess.BB_A8) != bool(new_rights & chess.BB_A8):
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_CASTLING_BASE + 3]

        # Handle en passant square changes
        old_ep, new_ep = move_info.previous_ep_square, self._board.ep_square
        if old_ep is not None:
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_EP_BASE + chess.square_file(old_ep)]
        if new_ep is not None:
            h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_EP_BASE + chess.square_file(new_ep)]

        # Toggle turn
        h ^= chess.polyglot.POLYGLOT_RANDOM_ARRAY[_ZOBRIST_TURN]
        return h

    def get_legal_moves_list(self) -> List[chess.Move]:
        """Get list of legal moves (cached)."""
        if self._cache.legal_moves is None:
            self._cache.legal_moves = list(self._board.legal_moves)
        return self._cache.legal_moves

    def is_check(self) -> bool:
        """Check if current side is in check (cached)."""
        if self._cache.is_check is None:
            self._cache.is_check = self._board.is_check()
        return self._cache.is_check

    def is_checkmate(self) -> bool:
        """Check if current position is checkmate (cached)."""
        if self._cache.is_checkmate is None:
            self._cache.is_checkmate = self._board.is_checkmate()
        return self._cache.is_checkmate

    def is_game_over(self) -> bool:
        """Check if game is over (cached)."""
        if self._cache.is_game_over is None:
            self._cache.is_game_over = self._board.is_game_over()
        return self._cache.is_game_over

    def has_non_pawn_material(self, color: Optional[bool] = None) -> bool:
        """Check if side has non-pawn material (cached)."""
        if color is None:
            color = self._board.turn
        if self._cache.has_non_pawn_material is None:
            self._cache.has_non_pawn_material = {}
        if color not in self._cache.has_non_pawn_material:
            result = any(self._board.pieces(pt, color)
                        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
            self._cache.has_non_pawn_material[color] = result
        return self._cache.has_non_pawn_material[color]

    # ==================== Move Info Pre-computation ====================

    def precompute_move_info(self) -> None:
        """
        Pre-compute is_capture, gives_check, victim_type, attacker_type
        for all legal moves. Much faster than computing individually.
        """
        if self._cache.move_is_capture is not None:
            return  # Already computed

        self._cache.move_is_capture = {}
        self._cache.move_gives_check = {}
        self._cache.move_victim_type = {}
        self._cache.move_attacker_type = {}

        legal_moves = self.get_legal_moves_list()
        occupied = self._board.occupied

        for move in legal_moves:
            to_sq = move.to_square
            from_sq = move.from_square

            # Fast capture detection using bitboard
            is_ep = self._board.is_en_passant(move)
            is_cap = bool(occupied & chess.BB_SQUARES[to_sq]) or is_ep
            self._cache.move_is_capture[move] = is_cap

            # gives_check is expensive - compute once
            self._cache.move_gives_check[move] = self._board.gives_check(move)

            # Victim type for MVV-LVA
            if is_cap:
                if is_ep:
                    self._cache.move_victim_type[move] = chess.PAWN
                else:
                    victim = self._board.piece_at(to_sq)
                    self._cache.move_victim_type[move] = victim.piece_type if victim else None
            else:
                self._cache.move_victim_type[move] = None

            # Attacker type
            attacker = self._board.piece_at(from_sq)
            self._cache.move_attacker_type[move] = attacker.piece_type if attacker else None

    def is_capture_cached(self, move: chess.Move) -> bool:
        """Get cached is_capture result."""
        if self._cache.move_is_capture is None:
            self.precompute_move_info()
        result = self._cache.move_is_capture.get(move)
        return result if result is not None else self._board.is_capture(move)

    def gives_check_cached(self, move: chess.Move) -> bool:
        """Get cached gives_check result."""
        if self._cache.move_gives_check is None:
            self.precompute_move_info()
        result = self._cache.move_gives_check.get(move)
        return result if result is not None else self._board.gives_check(move)

    def get_victim_type(self, move: chess.Move) -> Optional[int]:
        """Get piece type of captured piece (None if not a capture)."""
        if self._cache.move_victim_type is None:
            self.precompute_move_info()
        return self._cache.move_victim_type.get(move)

    def get_attacker_type(self, move: chess.Move) -> Optional[int]:
        """Get piece type of moving piece."""
        if self._cache.move_attacker_type is None:
            self.precompute_move_info()
        return self._cache.move_attacker_type.get(move)

    # ==================== Material Evaluation ====================

    def material_evaluation(self) -> int:
        """
        Get material + PST evaluation from side-to-move perspective.
        Uses incremental updates when possible.
        """
        if self._cache.material_evaluation is None:
            # Try incremental computation
            if len(self._cache_stack) > 1 and self._move_info_stack:
                parent_cache = self._cache_stack[-2]
                if parent_cache.material_evaluation is not None:
                    self._cache.material_evaluation = self._compute_incremental_material(
                        parent_cache.material_evaluation,
                        self._move_info_stack[-1],
                        parent_cache.is_endgame)
            # Fallback to full computation
            if self._cache.material_evaluation is None:
                self._cache.material_evaluation = self._compute_material_evaluation()
        return self._cache.material_evaluation

    def _is_endgame(self) -> bool:
        """Determine if position is an endgame (cached)."""
        if self._cache.is_endgame is None:
            wq = chess.popcount(self._board.pieces_mask(chess.QUEEN, chess.WHITE))
            bq = chess.popcount(self._board.pieces_mask(chess.QUEEN, chess.BLACK))
            if wq == 0 and bq == 0:
                self._cache.is_endgame = True
            else:
                self._cache.is_endgame = True
                for color in (chess.WHITE, chess.BLACK):
                    if chess.popcount(self._board.pieces_mask(chess.QUEEN, color)) > 0:
                        minors = chess.popcount(
                            self._board.pieces_mask(chess.KNIGHT, color) |
                            self._board.pieces_mask(chess.BISHOP, color))
                        rooks = chess.popcount(self._board.pieces_mask(chess.ROOK, color))
                        if rooks > 0 or minors > 1:
                            self._cache.is_endgame = False
                            break
        return self._cache.is_endgame

    def _compute_material_evaluation(self) -> int:
        """Compute full material + PST evaluation."""
        our_mat, their_mat = 0, 0
        our_color = self._board.turn
        is_eg = self._is_endgame()

        for sq in chess.SQUARES:
            piece = self._board.piece_at(sq)
            if piece is None:
                continue
            val = PIECE_VALUES[piece.piece_type] + _get_pst_value(
                piece.piece_type, sq, piece.color, is_eg)
            if piece.color == our_color:
                our_mat += val
            else:
                their_mat += val

        return our_mat - their_mat

    def _compute_incremental_material(self, parent_eval: int, move_info: _MoveInfo,
                                       parent_is_endgame: Optional[bool]) -> int:
        """Compute material evaluation incrementally from parent."""
        move = move_info.move
        is_eg = self._is_endgame()

        # If endgame status changed, recompute fully
        if parent_is_endgame is not None and parent_is_endgame != is_eg:
            return self._compute_material_evaluation()

        # Negate parent eval (side switched)
        new_eval = -parent_eval

        piece = self._board.piece_at(move.to_square)
        if piece is None:
            return self._compute_material_evaluation()

        moving_color = piece.color
        original_type = chess.PAWN if move.promotion else piece.piece_type

        # PST change for moving piece
        old_pst = _get_pst_value(original_type, move.from_square, moving_color, is_eg)
        new_pst = _get_pst_value(piece.piece_type, move.to_square, moving_color, is_eg)
        new_eval += old_pst - new_pst

        # Promotion material change
        if move.promotion:
            new_eval -= PIECE_VALUES[piece.piece_type] - PIECE_VALUES[chess.PAWN]

        # Captured piece value
        if move_info.captured_piece is not None:
            cap = move_info.captured_piece
            if move_info.was_en_passant:
                ep_sq = chess.square(chess.square_file(move.to_square),
                                     chess.square_rank(move.from_square))
                cap_pst = _get_pst_value(cap.piece_type, ep_sq, not moving_color, is_eg)
            else:
                cap_pst = _get_pst_value(cap.piece_type, move.to_square, not moving_color, is_eg)
            new_eval -= PIECE_VALUES[cap.piece_type] + cap_pst

        return new_eval