"""
CachedBoard - Efficient chess board wrapper with intelligent caching.

This module provides a chess.Board wrapper with caching for expensive computations,
incremental Zobrist hash updates, and piece-square table evaluation.

Supports both python-chess backend and fast C++ backend (chess_cpp).
The C++ backend provides 3-5x speedup in NPS.
"""

import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Try to import C++ backend, fall back to python-chess
try:
    import chess_cpp
    HAS_CPP_BACKEND = True
    print("✓ Using fast C++ chess backend (chess_cpp)")
except ImportError:
    HAS_CPP_BACKEND = False
    print("! C++ backend not available, using python-chess (slower)")
    import chess
    import chess.polyglot

# Import chess for compatibility types when needed
import chess
import chess.polyglot

# ========== Constants ==========

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

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


# ========== Move Adapter for C++ Backend ==========

class MoveAdapter:
    """
    Adapter to convert between python-chess Move and chess_cpp Move.
    Provides a unified interface regardless of backend.
    """
    
    @staticmethod
    def to_chess_move(cpp_move: Any) -> chess.Move:
        """Convert chess_cpp.Move to chess.Move"""
        if not HAS_CPP_BACKEND:
            return cpp_move
        return chess.Move(
            cpp_move.from_square,
            cpp_move.to_square,
            cpp_move.promotion if cpp_move.promotion > 0 else None
        )
    
    @staticmethod
    def from_chess_move(py_move: chess.Move) -> Any:
        """Convert chess.Move to chess_cpp.Move"""
        if not HAS_CPP_BACKEND:
            return py_move
        return chess_cpp.Move(
            py_move.from_square,
            py_move.to_square,
            py_move.promotion if py_move.promotion else 0
        )


# ========== CachedBoard ==========

class CachedBoard:
    """
    Chess board wrapper with caching for expensive computations.

    Uses composition to wrap either chess.Board (Python) or chess_cpp.Board (C++),
    automatically selecting the fastest available backend.
    
    The C++ backend provides 3-5x speedup in move generation and game state queries.
    """

    __slots__ = ('_board', '_py_board', '_py_board_dirty', '_cache_stack',
                 '_move_info_stack', '_move_stack', '_use_cpp', '_initial_fen',
                 '_cpp_stack_dirty', '_hash_history')

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN):
        """Initialize board from FEN string. Pass None for empty board."""
        self._cache_stack: List[_CacheState] = [_CacheState()]
        self._move_info_stack: List[_MoveInfo] = []
        self._move_stack: List[chess.Move] = []  # Track moves for repetition
        self._use_cpp = HAS_CPP_BACKEND
        self._py_board: Optional[chess.Board] = None  # Lazy initialization (rarely needed now)
        self._py_board_dirty = True  # Needs rebuild before use
        self._cpp_stack_dirty = False  # True when C++ board's internal stack is out of sync
        self._hash_history: List[int] = []  # Track position hashes for repetition detection

        # Store initial FEN for lazy py_board rebuild
        # This is critical: when rebuilding the Python board for repetition detection,
        # we need to start from the ACTUAL initial position, not the starting position
        self._initial_fen = fen if fen is not None else "8/8/8/8/8/8/8/8 w - - 0 1"

        if self._use_cpp:
            if fen is None:
                # Empty board - use a known empty FEN
                self._board = chess_cpp.Board("8/8/8/8/8/8/8/8 w - - 0 1")
            else:
                self._board = chess_cpp.Board(fen)
            # Record initial position hash
            self._hash_history.append(self._board.polyglot_hash())
        else:
            self._board = chess.Board(fen) if fen else chess.Board(None)
            self._py_board = self._board  # Same object when using python-chess
            self._py_board_dirty = False
            self._hash_history.append(chess.polyglot.zobrist_hash(self._board))

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
        # C++ returns uint64_t, ensure we return int for compatibility
        return int(self._board.castling_rights)

    @property
    def ep_square(self) -> Optional[int]:
        ep = self._board.ep_square
        return ep if ep >= 0 else None

    @property
    def move_stack(self) -> List[chess.Move]:
        return self._move_stack

    @property
    def legal_moves(self):
        """Iterator over legal moves (for compatibility with chess.Board)."""
        return iter(self.get_legal_moves_list())

    # ==================== Core Methods ====================

    def push(self, move: chess.Move) -> None:
        """Make a move on the board."""
        # Handle null moves specially (used in null-move pruning)
        is_null_move = (move.from_square == move.to_square == 0 and move.promotion is None)

        if is_null_move:
            # For null moves, we need to flip the turn without moving pieces
            # Store minimal info for pop()
            move_info = _MoveInfo(
                move=move,
                captured_piece=None,
                was_en_passant=False,
                was_castling=False,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            if self._use_cpp:
                # For C++ backend: we can't push null moves directly
                # Instead, we'll modify the FEN to flip the turn
                current_fen = self._board.fen()
                parts = current_fen.split(' ')
                # Flip turn
                parts[1] = 'b' if parts[1] == 'w' else 'w'
                # Clear en passant square (null move clears it)
                parts[3] = '-'
                new_fen = ' '.join(parts)
                self._board.set_fen(new_fen)
                self._py_board_dirty = True
                self._cpp_stack_dirty = True  # C++ board's internal stack is now corrupted
            else:
                self._board.push(move)
        else:
            # Regular move handling
            # Capture move info before pushing
            move_info = _MoveInfo(
                move=move,
                captured_piece=self._get_captured_piece(move),
                was_en_passant=self.is_en_passant(move),
                was_castling=self.is_castling(move),
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            if self._use_cpp:
                if self._cpp_stack_dirty:
                    # Stack is dirty, use FEN manipulation instead of push
                    # First make the move to get the resulting position
                    temp_board = chess.Board(self._board.fen())
                    temp_board.push(move)
                    self._board.set_fen(temp_board.fen())
                else:
                    cpp_move = MoveAdapter.from_chess_move(move)
                    self._board.push(cpp_move)
                self._py_board_dirty = True
            else:
                self._board.push(move)

        self._move_stack.append(move)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(_CacheState())

        # Record position hash for repetition detection
        self._hash_history.append(self.zobrist_hash())

    def pop(self) -> chess.Move:
        """Unmake the last move."""
        if not self._move_stack:
            raise IndexError("pop from empty move stack")

        move = self._move_stack[-1]
        move_info = self._move_info_stack[-1] if self._move_info_stack else None
        is_null_move = (move.from_square == move.to_square == 0 and move.promotion is None)

        # Remove hash from history
        if self._hash_history:
            self._hash_history.pop()

        if self._use_cpp:
            if self._cpp_stack_dirty or is_null_move:
                # Stack is dirty or this is a null move - use FEN restoration
                # We need to reconstruct the position before this move

                # Remove this move from our stack first (temporarily)
                self._move_stack.pop()
                if len(self._cache_stack) > 1:
                    self._cache_stack.pop()
                if self._move_info_stack:
                    self._move_info_stack.pop()

                # Rebuild position from initial FEN + remaining moves
                self._board.set_fen(self._initial_fen)
                for m in self._move_stack:
                    m_is_null = (m.from_square == m.to_square == 0 and m.promotion is None)
                    if m_is_null:
                        fen = self._board.fen()
                        parts = fen.split(' ')
                        parts[1] = 'b' if parts[1] == 'w' else 'w'
                        parts[3] = '-'
                        self._board.set_fen(' '.join(parts))
                    else:
                        cpp_m = MoveAdapter.from_chess_move(m)
                        self._board.push(cpp_m)

                # Check if stack is still dirty (any null moves remain?)
                self._cpp_stack_dirty = any(
                    m.from_square == m.to_square == 0 and m.promotion is None
                    for m in self._move_stack
                )

                self._py_board_dirty = True
                return move
            else:
                cpp_move = self._board.pop()
                move = MoveAdapter.to_chess_move(cpp_move)
                self._py_board_dirty = True
        else:
            move = self._board.pop()

        self._move_stack.pop()
        if len(self._cache_stack) > 1:
            self._cache_stack.pop()
        if self._move_info_stack:
            self._move_info_stack.pop()
        return move

    def _get_captured_piece(self, move: chess.Move) -> Optional[chess.Piece]:
        """Get the piece captured by a move (before the move is made)."""
        if self.is_en_passant(move):
            return chess.Piece(chess.PAWN, not self.turn)
        piece_opt = self._board.piece_at(move.to_square)
        if self._use_cpp:
            if piece_opt is None:
                return None
            return chess.Piece(piece_opt.piece_type, piece_opt.color)
        return piece_opt

    def copy(self, stack: bool = True) -> "CachedBoard":
        """Create a copy of the board."""
        if stack and self._move_stack:
            # Replay all moves from initial position to build up state
            board = CachedBoard(self._initial_fen)
            for move in self._move_stack:
                board.push(move)
        else:
            board = CachedBoard(self.fen())
        return board

    def set_fen(self, fen: str) -> None:
        """Set position from FEN string."""
        self._board.set_fen(fen)
        self._cache_stack = [_CacheState()]
        self._move_info_stack = []
        self._move_stack = []
        self._initial_fen = fen  # Track new initial position
        # Reset hash history with new position
        self._hash_history = [self.zobrist_hash()]
        if self._use_cpp:
            self._py_board_dirty = True
            self._cpp_stack_dirty = False  # Fresh start, stack is clean

    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()

    # ==================== Delegated Methods ====================

    def piece_at(self, square: int) -> Optional[chess.Piece]:
        piece_opt = self._board.piece_at(square)
        if self._use_cpp:
            if piece_opt is None:
                return None
            return chess.Piece(piece_opt.piece_type, piece_opt.color)
        return piece_opt

    def king(self, color: bool) -> Optional[int]:
        result = self._board.king(color)
        if self._use_cpp:
            return result if result >= 0 else None
        return result

    def san(self, move: chess.Move) -> str:
        if self._use_cpp:
            cpp_move = MoveAdapter.from_chess_move(move)
            return self._board.san(cpp_move)
        return self._board.san(move)

    def parse_san(self, san: str) -> chess.Move:
        if self._use_cpp:
            cpp_move = self._board.parse_san(san)
            return MoveAdapter.to_chess_move(cpp_move)
        return self._board.parse_san(san)

    def is_en_passant(self, move: chess.Move) -> bool:
        if self._use_cpp:
            cpp_move = MoveAdapter.from_chess_move(move)
            return self._board.is_en_passant(cpp_move)
        return self._board.is_en_passant(move)

    def is_castling(self, move: chess.Move) -> bool:
        if self._use_cpp:
            cpp_move = MoveAdapter.from_chess_move(move)
            return self._board.is_castling(cpp_move)
        return self._board.is_castling(move)

    def is_capture(self, move: chess.Move) -> bool:
        if self._use_cpp:
            cpp_move = MoveAdapter.from_chess_move(move)
            return self._board.is_capture(cpp_move)
        return self._board.is_capture(move)

    def _ensure_py_board(self) -> chess.Board:
        """Ensure _py_board is synchronized for repetition detection.

        This is optimized to keep the Python board incrementally in sync,
        avoiding expensive full rebuilds on every call.
        """
        if not self._use_cpp:
            return self._board

        if self._py_board is None:
            # First time - build from scratch
            self._py_board = chess.Board(self._initial_fen)
            for move in self._move_stack:
                self._py_board.push(move)
            self._py_board_dirty = False
            return self._py_board

        if not self._py_board_dirty:
            return self._py_board

        # Sync incrementally by comparing move stacks
        py_len = len(self._py_board.move_stack)
        our_len = len(self._move_stack)

        # Find common prefix length
        common_len = 0
        min_len = min(py_len, our_len)
        for i in range(min_len):
            if self._py_board.move_stack[i] != self._move_stack[i]:
                break
            common_len = i + 1
        else:
            common_len = min_len

        # Pop excess moves from py_board
        while len(self._py_board.move_stack) > common_len:
            self._py_board.pop()

        # Push missing moves
        for move in self._move_stack[common_len:]:
            self._py_board.push(move)

        self._py_board_dirty = False
        return self._py_board

    def is_repetition(self, count: int = 3) -> bool:
        """Check for repetition using hash history (fast, no py_board needed)."""
        if not self._hash_history:
            return False

        current_hash = self._hash_history[-1]
        occurrences = 0

        # Count how many times this position has occurred
        # Only need to check positions where the same side was to move
        # (every other position in the history)
        for i in range(len(self._hash_history) - 1, -1, -2):
            if self._hash_history[i] == current_hash:
                occurrences += 1
                if occurrences >= count:
                    return True

        return False

    def can_claim_fifty_moves(self) -> bool:
        return self._board.can_claim_fifty_moves()

    def ply(self) -> int:
        return self._board.ply()

    def pieces(self, piece_type: int, color: bool) -> chess.SquareSet:
        """Get square set of pieces of given type and color."""
        if self._use_cpp:
            mask = self._board.pieces_mask(piece_type, color)
            return chess.SquareSet(mask)
        return self._board.pieces(piece_type, color)

    def pieces_mask(self, piece_type: int, color: bool) -> int:
        return self._board.pieces_mask(piece_type, color)

    # ==================== Cached Methods ====================

    def zobrist_hash(self) -> int:
        """Get Polyglot-compatible Zobrist hash (cached)."""
        if self._cache.zobrist_hash is None:
            if self._use_cpp:
                # Use C++ Polyglot implementation
                self._cache.zobrist_hash = self._board.polyglot_hash()
            else:
                # Use python-chess
                self._cache.zobrist_hash = chess.polyglot.zobrist_hash(self._board)
        return self._cache.zobrist_hash

    def get_legal_moves_list(self) -> List[chess.Move]:
        """Get list of legal moves (cached)."""
        if self._cache.legal_moves is None:
            if self._use_cpp:
                cpp_moves = self._board.legal_moves()
                self._cache.legal_moves = [MoveAdapter.to_chess_move(m) for m in cpp_moves]
            else:
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
            color = self.turn
        if self._cache.has_non_pawn_material is None:
            self._cache.has_non_pawn_material = {}
        if color not in self._cache.has_non_pawn_material:
            result = any(self.pieces_mask(pt, color)
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
        occupied = self.occupied

        for move in legal_moves:
            to_sq = move.to_square
            from_sq = move.from_square

            # Fast capture detection using bitboard
            is_ep = self.is_en_passant(move)
            is_cap = bool(occupied & chess.BB_SQUARES[to_sq]) or is_ep
            self._cache.move_is_capture[move] = is_cap

            # gives_check is expensive - compute once
            if self._use_cpp:
                cpp_move = MoveAdapter.from_chess_move(move)
                self._cache.move_gives_check[move] = self._board.gives_check(cpp_move)
            else:
                self._cache.move_gives_check[move] = self._board.gives_check(move)

            # Victim type for MVV-LVA
            if is_cap:
                if is_ep:
                    self._cache.move_victim_type[move] = chess.PAWN
                else:
                    victim = self.piece_at(to_sq)
                    self._cache.move_victim_type[move] = victim.piece_type if victim else None
            else:
                self._cache.move_victim_type[move] = None

            # Attacker type
            attacker = self.piece_at(from_sq)
            self._cache.move_attacker_type[move] = attacker.piece_type if attacker else None

    def is_capture_cached(self, move: chess.Move) -> bool:
        """Get cached is_capture result."""
        if self._cache.move_is_capture is None:
            self.precompute_move_info()
        result = self._cache.move_is_capture.get(move)
        return result if result is not None else self.is_capture(move)

    def gives_check_cached(self, move: chess.Move) -> bool:
        """Get cached gives_check result."""
        if self._cache.move_gives_check is None:
            self.precompute_move_info()
        result = self._cache.move_gives_check.get(move)
        if result is not None:
            return result
        if self._use_cpp:
            cpp_move = MoveAdapter.from_chess_move(move)
            return self._board.gives_check(cpp_move)
        return self._board.gives_check(move)

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
            if self._use_cpp:
                wq = chess_cpp.popcount(self.pieces_mask(chess.QUEEN, chess.WHITE))
                bq = chess_cpp.popcount(self.pieces_mask(chess.QUEEN, chess.BLACK))
            else:
                wq = chess.popcount(self.pieces_mask(chess.QUEEN, chess.WHITE))
                bq = chess.popcount(self.pieces_mask(chess.QUEEN, chess.BLACK))

            if wq == 0 and bq == 0:
                self._cache.is_endgame = True
            else:
                self._cache.is_endgame = True
                for color in (chess.WHITE, chess.BLACK):
                    if self._use_cpp:
                        has_queen = chess_cpp.popcount(self.pieces_mask(chess.QUEEN, color)) > 0
                    else:
                        has_queen = chess.popcount(self.pieces_mask(chess.QUEEN, color)) > 0

                    if has_queen:
                        if self._use_cpp:
                            minors = chess_cpp.popcount(
                                self.pieces_mask(chess.KNIGHT, color) |
                                self.pieces_mask(chess.BISHOP, color))
                            rooks = chess_cpp.popcount(self.pieces_mask(chess.ROOK, color))
                        else:
                            minors = chess.popcount(
                                self.pieces_mask(chess.KNIGHT, color) |
                                self.pieces_mask(chess.BISHOP, color))
                            rooks = chess.popcount(self.pieces_mask(chess.ROOK, color))

                        if rooks > 0 or minors > 1:
                            self._cache.is_endgame = False
                            break
        return self._cache.is_endgame

    def _compute_material_evaluation(self) -> int:
        """Compute full material + PST evaluation."""
        our_mat, their_mat = 0, 0
        our_color = self.turn
        is_eg = self._is_endgame()

        for sq in chess.SQUARES:
            piece = self.piece_at(sq)
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

        piece = self.piece_at(move.to_square)
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