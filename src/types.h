#pragma once

#include <cstdint>
#include <cstdlib>

using Bitboard = uint64_t;
using Key     = uint64_t;

enum File : int { FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1 = 0, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

constexpr int idx(int file, int rank) { return rank * 8 + file; }

enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NB = 64, SQ_NONE
};

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB, NO_PIECE_TYPE };

enum Piece : int {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16, NO_PIECE
};

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) | pt); }
constexpr Color    color_of(Piece p) { return Color((p >> 3) & 1); }
constexpr PieceType type_of(Piece p) { return PieceType(p & 7); }

constexpr int SQ(int rank, int file) { return rank * 8 + file; }

enum CastlingRights : uint8_t {
    WHITE_KING_SIDE  = 1,
    WHITE_QUEEN_SIDE = 2,
    BLACK_KING_SIDE  = 4,
    BLACK_QUEEN_SIDE = 8,
    ANY_CASTLING     = 15
};

enum MoveFlags : uint16_t {
    QUIET           = 0b0000,
    DOUBLE_PAWN     = 0b0001,
    KING_CASTLE     = 0b0010,
    QUEEN_CASTLE    = 0b0011,
    CAPTURE         = 0b0100,
    EP_CAPTURE      = 0b0101,
    PROMOTION       = 0b1000,
    KNIGHT_PROMO    = 0b1000,
    BISHOP_PROMO    = 0b1001,
    ROOK_PROMO      = 0b1010,
    QUEEN_PROMO     = 0b1011,
    KNIGHT_PROMO_CAP= 0b1100,
    BISHOP_PROMO_CAP= 0b1101,
    ROOK_PROMO_CAP  = 0b1110,
    QUEEN_PROMO_CAP = 0b1111,
    FLAG_MASK       = 0b1111
};

enum Value : int { VALUE_MATE = 32000, VALUE_INFINITE = 32001, VALUE_NONE = 32002 };

constexpr int FILE_OF(int s) { return s & 7; }
constexpr int RANK_OF(int s) { return s >> 3; }
constexpr int relative_rank(Color c, int s) { return c == WHITE ? RANK_OF(s) : 7 - RANK_OF(s); }
