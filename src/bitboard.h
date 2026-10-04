#pragma once

#include "types.h"

namespace bb {

constexpr Bitboard FILE_BB[8] = {
    0x0101010101010101ULL, 0x0202020202020202ULL,
    0x0404040404040404ULL, 0x0808080808080808ULL,
    0x1010101010101010ULL, 0x2020202020202020ULL,
    0x4040404040404040ULL, 0x8080808080808080ULL,
};

constexpr Bitboard RANK_BB[8] = {
    0xFFULL, 0xFF00ULL, 0xFF0000ULL, 0xFF000000ULL,
    0xFF00000000ULL, 0xFF0000000000ULL, 0xFF000000000000ULL, 0xFF00000000000000ULL,
};

void init();

extern Bitboard pawn_attacks[COLOR_NB][SQ_NB];
extern Bitboard knight_attacks[SQ_NB];
extern Bitboard king_attacks[SQ_NB];

Bitboard pseudo_attacks(PieceType pt, Square s, Bitboard occ);

// Self-test: verifies every attack table (pawns/knights/king) and the sliding
// attack lookup (PEXT table or fallback) against independent reference
// computations. Returns true when all checks pass.
bool self_test(long long& checks, long long& fails);

}  // namespace bb
