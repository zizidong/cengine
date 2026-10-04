#pragma once

#include "types.h"

#include <cstdint>

// Move encoding (32 bits)
// bits 0-5   : from square
// bits 6-11  : to square
// bits 12-15 : flags (MoveFlags)
class Move {
public:
    constexpr Move() : m(0) {}
    constexpr Move(Square from, Square to, uint16_t flags) : m((flags << 12) | (to << 6) | from) {}
    constexpr Move(uint32_t raw) : m(raw) {}

    constexpr Square from() const { return Square(m & 0x3F); }
    constexpr Square to()   const { return Square((m >> 6) & 0x3F); }
    constexpr uint16_t flags() const { return uint16_t((m >> 12) & 0xF); }
    constexpr uint32_t raw() const { return m; }

    constexpr bool is_capture() const { return flags() & CAPTURE; }
    constexpr bool is_promotion() const { return flags() & PROMOTION; }
    constexpr bool is_castle() const { return flags() == KING_CASTLE || flags() == QUEEN_CASTLE; }
    constexpr bool is_ep() const { return flags() == EP_CAPTURE; }
    constexpr bool is_quiet() const { return flags() == QUIET || flags() == DOUBLE_PAWN; }

    constexpr PieceType promo_type() const {
        switch (flags() & 0b0011) {
            case 0b00: return KNIGHT;
            case 0b01: return BISHOP;
            case 0b10: return ROOK;
            default:   return QUEEN;
        }
    }

    constexpr bool operator==(const Move& o) const { return m == o.m; }
    constexpr bool operator!=(const Move& o) const { return m != o.m; }

    static constexpr Move none() { return Move(0); }

private:
    uint32_t m;
};

struct HistoryEntry {
    Move move;
    int captured;
    uint8_t castling;
    Square ep;
    uint8_t fifty;
    Key key;
};

class Board {
public:
    Board();

    void set_fen(const char* fen);
    void reset();
    void to_fen(char* out) const;

    Piece piece_on(Square s) const { return board[s]; }
    PieceType type_on(Square s) const { return type_of(board[s]); }
    Color side_to_move() const { return stm; }

    Bitboard pieces(PieceType pt) const { return by_type[pt]; }
    Bitboard pieces(Color c) const { return by_color[c]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_type[pt] & by_color[c]; }

    Bitboard occupancy() const { return occ; }

    Key key() const { return state_key; }
    Square ep_square() const { return ep; }
    int castling() const { return castling_rights; }
    Square king_square(Color c) const { return king_sq[c]; }

    bool gives_check(Move m, Bitboard occ) const;
    bool makes_pseudolegal(Move m) const;
    Color non_us() const { return ~stm; }

    void make_move(Move m);
    void unmake_move();

    Bitboard attackers_to(Square s, Bitboard occ) const;
    Bitboard checkers() const;
    Bitboard checkers(Color c) const;
    bool in_check() const { return checkers() != 0; }

    // true if the player who just moved (now non_us) is in check -> illegal move
    bool moved_into_check() const { return checkers(non_us()) != 0; }

    // Our pieces that are pinned to our king (computed once per node).
    Bitboard pinned_pieces(Color c) const;

    // Fast pre-move legality filter using the pinned set. Ep moves are not
    // decided here (the caller still verifies them with moved_into_check()).
    bool legal_precheck(Move m, Bitboard pinned) const;

    // Static exchange evaluation: true if the capture sequence starting with m
    // gains at least `threshold` (in centipawns).
    bool see_ge(Move m, int threshold) const;
    bool is_repetition() const;
    bool is_insufficient_material() const;

    Piece board[SQ_NB];
    Bitboard by_type[PIECE_TYPE_NB];
    Bitboard by_color[COLOR_NB];
    Bitboard occ;  // cached occupancy, kept in sync by make/unmake/set_fen
    Square king_sq[COLOR_NB];
    Color stm;
    int castling_rights;
    Square ep;
    int fifty;
    int ply;
    Key state_key;
    int history_ply;
    HistoryEntry history[512];
};