#include "zobrist.h"

namespace zobrist {

Key psq[PIECE_NB][SQ_NB];
Key castling[16];
Key ep_file[8];
Key side;

namespace {

uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

uint64_t rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

}  // namespace

void init() {
    for (int p = 0; p < PIECE_NB; ++p)
        for (int s = 0; s < SQ_NB; ++s)
            psq[p][s] = rng();
    for (int c = 0; c < 16; ++c)
        castling[c] = rng();
    for (int f = 0; f < 8; ++f)
        ep_file[f] = rng();
    side = rng();
}

}  // namespace zobrist