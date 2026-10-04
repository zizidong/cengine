#pragma once

#include "types.h"

namespace attack {

Bitboard pawn_attacks(Bitboard pawns, Color c);
Bitboard piece_attacks(PieceType pt, Square s, Bitboard occ);
Bitboard attacks_by(Color c, Bitboard occ);
Bitboard attackers_to(Square s, Bitboard occ);

}  // namespace attack