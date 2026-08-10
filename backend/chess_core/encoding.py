"""AlphaZero-style board and action encoding for chess."""

from __future__ import annotations

import chess
import numpy as np

# Classic AlphaZero action space: 8x8x73
# N, NE, E, SE, S, SW, W, NW in (rank_delta, file_delta)
QUEEN_DIRS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]
KNIGHT_DELTAS = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_QUEEN_MOVES = 56  # 8 directions * 7 distances
NUM_KNIGHT_MOVES = 8
NUM_UNDERPROMOTIONS = 9  # 3 piece types * 3 left/center/right
PLANES_PER_SQUARE = NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_UNDERPROMOTIONS  # 73
ACTION_SIZE = 64 * PLANES_PER_SQUARE  # 4672

PIECE_PLANES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def _sq_to_rc(square: int) -> tuple[int, int]:
    return divmod(square, 8)  # rank, file


def _rc_to_sq(rank: int, file: int) -> int:
    return rank * 8 + file


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode board as (19, 8, 8) float32 planes from the side-to-move perspective.

    Planes:
      0-5   our pieces (P,N,B,R,Q,K)
      6-11  opponent pieces
      12    side-to-move constant (1 if white to move else 0) — after flip always 1 for us
      13-16 castling rights (ours KQ, opp KQ)
      17    en-passant file (one-hot on file if available)
      18    move-count / 100 (normalized halfmove-ish progress proxy via fullmove)
    """
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    we_are_white = board.turn == chess.WHITE

    def map_sq(sq: int) -> int:
        return sq if we_are_white else chess.square_mirror(sq)

    for piece_type_idx, piece_type in enumerate(PIECE_PLANES):
        for color_is_white in (True, False):
            ours = color_is_white == we_are_white
            plane = piece_type_idx if ours else piece_type_idx + 6
            color = chess.WHITE if color_is_white else chess.BLACK
            for sq in board.pieces(piece_type, color):
                r, f = _sq_to_rc(map_sq(sq))
                planes[plane, r, f] = 1.0

    # Side indicator kept for compatibility with fixed architectures
    planes[12, :, :] = 1.0

    # Castling from our perspective
    if we_are_white:
        ours_k, ours_q = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE)
        opp_k, opp_q = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)
    else:
        ours_k, ours_q = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)
        opp_k, opp_q = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE)

    if ours_k:
        planes[13, :, :] = 1.0
    if ours_q:
        planes[14, :, :] = 1.0
    if opp_k:
        planes[15, :, :] = 1.0
    if opp_q:
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        ep = map_sq(board.ep_square)
        _, f = _sq_to_rc(ep)
        planes[17, :, f] = 1.0

    planes[18, :, :] = min(board.fullmove_number, 100) / 100.0
    return planes


def _queen_plane(dr: int, df: int) -> int | None:
    if dr == 0 and df == 0:
        return None
    if dr != 0 and df != 0 and abs(dr) != abs(df):
        return None
    if dr != 0 and df == 0:
        dist = abs(dr)
        direction = 0 if dr > 0 else 4  # N / S
    elif dr == 0 and df != 0:
        dist = abs(df)
        direction = 2 if df > 0 else 6  # E / W
    else:
        dist = abs(dr)
        if dr > 0 and df > 0:
            direction = 1  # NE
        elif dr < 0 and df > 0:
            direction = 3  # SE
        elif dr < 0 and df < 0:
            direction = 5  # SW
        else:
            direction = 7  # NW
    if not 1 <= dist <= 7:
        return None
    return direction * 7 + (dist - 1)


def _knight_plane(dr: int, df: int) -> int | None:
    try:
        return NUM_QUEEN_MOVES + KNIGHT_DELTAS.index((dr, df))
    except ValueError:
        return None


def _underpromotion_plane(move: chess.Move, we_are_white: bool) -> int | None:
    if move.promotion is None or move.promotion == chess.QUEEN:
        return None
    try:
        promo_idx = UNDERPROMOTIONS.index(move.promotion)
    except ValueError:
        return None

    from_sq = move.from_square if we_are_white else chess.square_mirror(move.from_square)
    to_sq = move.to_square if we_are_white else chess.square_mirror(move.to_square)
    fr, ff = _sq_to_rc(from_sq)
    tr, tf = _sq_to_rc(to_sq)
    df = tf - ff
    # left / straight / right from our perspective (increasing file = right)
    if df == -1:
        file_idx = 0
    elif df == 0:
        file_idx = 1
    elif df == 1:
        file_idx = 2
    else:
        return None
    return NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + promo_idx * 3 + file_idx


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Map a legal move to an AlphaZero action index in [0, 4672)."""
    we_are_white = board.turn == chess.WHITE
    from_sq = move.from_square if we_are_white else chess.square_mirror(move.from_square)
    to_sq = move.to_square if we_are_white else chess.square_mirror(move.to_square)
    fr, ff = _sq_to_rc(from_sq)
    tr, tf = _sq_to_rc(to_sq)
    dr, df = tr - fr, tf - ff

    plane = _underpromotion_plane(move, we_are_white)
    if plane is None:
        plane = _knight_plane(dr, df)
    if plane is None:
        plane = _queen_plane(dr, df)
    if plane is None:
        raise ValueError(f"Cannot encode move {move.uci()} on {board.fen()}")
    return from_sq * PLANES_PER_SQUARE + plane


def decode_move(board: chess.Board, action: int) -> chess.Move | None:
    """Decode action index into a chess.Move if it is legal on this board."""
    from_sq_view = action // PLANES_PER_SQUARE
    plane = action % PLANES_PER_SQUARE
    we_are_white = board.turn == chess.WHITE

    fr, ff = _sq_to_rc(from_sq_view)

    if plane < NUM_QUEEN_MOVES:
        direction = plane // 7
        dist = (plane % 7) + 1
        dr, df = QUEEN_DIRS[direction]
        tr, tf = fr + dr * dist, ff + df * dist
        if not (0 <= tr < 8 and 0 <= tf < 8):
            return None
        to_sq_view = _rc_to_sq(tr, tf)
        promotion = None
        # Queen promotion when pawn reaches last rank via queen-move encoding
        from_real = from_sq_view if we_are_white else chess.square_mirror(from_sq_view)
        piece = board.piece_at(from_real)
        if piece is not None and piece.piece_type == chess.PAWN and tr in (0, 7):
            promotion = chess.QUEEN
    elif plane < NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES:
        dr, df = KNIGHT_DELTAS[plane - NUM_QUEEN_MOVES]
        tr, tf = fr + dr, ff + df
        if not (0 <= tr < 8 and 0 <= tf < 8):
            return None
        to_sq_view = _rc_to_sq(tr, tf)
        promotion = None
    else:
        up = plane - NUM_QUEEN_MOVES - NUM_KNIGHT_MOVES
        promo_idx, file_idx = divmod(up, 3)
        promotion = UNDERPROMOTIONS[promo_idx]
        df = file_idx - 1
        # Pawn moves "forward" in our view: increasing rank for white perspective
        tr, tf = fr + 1, ff + df
        if not (0 <= tr < 8 and 0 <= tf < 8):
            return None
        to_sq_view = _rc_to_sq(tr, tf)

    from_sq = from_sq_view if we_are_white else chess.square_mirror(from_sq_view)
    to_sq = to_sq_view if we_are_white else chess.square_mirror(to_sq_view)
    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move in board.legal_moves:
        return move
    # Castling is encoded as king two-file move; python-chess expects that.
    return None


def policy_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask over ACTION_SIZE for legal moves."""
    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        try:
            mask[encode_move(board, move)] = True
        except ValueError:
            continue
    return mask
