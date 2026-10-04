#include "attack.h"
#include "bitboard.h"

namespace attack {

Bitboard pawn_attacks(Bitboard pawns, Color c) {
    if (c == WHITE)
        return ((pawns << 7) & ~bb::FILE_BB[FILE_H]) | ((pawns << 9) & ~bb::FILE_BB[FILE_A]);
    else
        return ((pawns >> 7) & ~bb::FILE_BB[FILE_A]) | ((pawns >> 9) & ~bb::FILE_BB[FILE_H]);
}

Bitboard piece_attacks(PieceType pt, Square s, Bitboard occ) {
    return bb::pseudo_attacks(pt, s, occ);
}

}  // namespace attack
