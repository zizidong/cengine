#include "bitboard.h"

#include <cstdlib>
#include <initializer_list>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace bb {

Bitboard pawn_attacks[COLOR_NB][SQ_NB];
Bitboard knight_attacks[SQ_NB];
Bitboard king_attacks[SQ_NB];

namespace {

const int bishop_dirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
const int rook_dirs[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};

// Ray-based sliding attacks (used to build the lookup tables and as the
// portable fallback when BMI2 is unavailable).
[[nodiscard]] Bitboard slide_attacks(int sq, Bitboard occ, const int dirs[4][2]) {
    Bitboard attacks = 0;
    int f = FILE_OF(sq), r = RANK_OF(sq);
    for (int d = 0; d < 4; ++d) {
        int df = dirs[d][0], dr = dirs[d][1];
        int cf = f + df, cr = r + dr;
        while (cf >= 0 && cf < 8 && cr >= 0 && cr < 8) {
            int s = SQ(cr, cf);
            attacks |= 1ULL << s;
            if (occ & (1ULL << s)) break;
            cf += df; cr += dr;
        }
    }
    return attacks;
}

// Relevant-occupancy masks (exclude the board edges and the square itself).
[[nodiscard]] Bitboard rook_mask(int sq) {
    Bitboard mask = 0;
    int f = FILE_OF(sq), r = RANK_OF(sq);
    for (int cf = f + 1; cf <= 6; ++cf) mask |= 1ULL << SQ(r, cf);
    for (int cf = f - 1; cf >= 1; --cf) mask |= 1ULL << SQ(r, cf);
    for (int cr = r + 1; cr <= 6; ++cr) mask |= 1ULL << SQ(cr, f);
    for (int cr = r - 1; cr >= 1; --cr) mask |= 1ULL << SQ(cr, f);
    return mask;
}

[[nodiscard]] Bitboard bishop_mask(int sq) {
    Bitboard mask = 0;
    int f = FILE_OF(sq), r = RANK_OF(sq);
    for (int cf = f + 1, cr = r + 1; cf <= 6 && cr <= 6; ++cf, ++cr) mask |= 1ULL << SQ(cr, cf);
    for (int cf = f + 1, cr = r - 1; cf <= 6 && cr >= 1; ++cf, --cr) mask |= 1ULL << SQ(cr, cf);
    for (int cf = f - 1, cr = r + 1; cf >= 1 && cr <= 6; --cf, ++cr) mask |= 1ULL << SQ(cr, cf);
    for (int cf = f - 1, cr = r - 1; cf >= 1 && cr >= 1; --cf, --cr) mask |= 1ULL << SQ(cr, cf);
    return mask;
}

#if defined(__BMI2__)
Bitboard rook_mask_tbl[64];
Bitboard bishop_mask_tbl[64];
Bitboard rook_table[64 * 4096];
Bitboard bishop_table[64 * 512];

inline Bitboard rook_attacks(Square s, Bitboard occ) {
    return rook_table[s * 4096 + _pext_u64(occ, rook_mask_tbl[s])];
}
inline Bitboard bishop_attacks(Square s, Bitboard occ) {
    return bishop_table[s * 512 + _pext_u64(occ, bishop_mask_tbl[s])];
}
#else
inline Bitboard rook_attacks(Square s, Bitboard occ) { return slide_attacks(s, occ, rook_dirs); }
inline Bitboard bishop_attacks(Square s, Bitboard occ) { return slide_attacks(s, occ, bishop_dirs); }
#endif

}  // namespace

void init() {
    for (int s = 0; s < SQ_NB; ++s) {
        Bitboard b = 1ULL << s;
        int f = FILE_OF(s), r = RANK_OF(s);

        pawn_attacks[WHITE][s] = ((b << 7) & ~FILE_BB[FILE_H]) | ((b << 9) & ~FILE_BB[FILE_A]);
        pawn_attacks[BLACK][s] = ((b >> 7) & ~FILE_BB[FILE_A]) | ((b >> 9) & ~FILE_BB[FILE_H]);

        knight_attacks[s] = 0;
        for (int df : {-2,-1,1,2}) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            int rem = 3 - std::abs(df);
            for (int dr : {-rem, rem}) {
                if (dr == 0) continue;
                int nr = r + dr;
                if (nr >= 0 && nr <= 7) knight_attacks[s] |= 1ULL << SQ(nr, nf);
            }
        }

        king_attacks[s] = 0;
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) continue;
                int nf = f + df, nr = r + dr;
                if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7)
                    king_attacks[s] |= 1ULL << SQ(nr, nf);
            }
    }

#if defined(__BMI2__)
    for (int s = 0; s < 64; ++s) {
        const Bitboard rmask = rook_mask(s);
        const Bitboard bmask = bishop_mask(s);
        rook_mask_tbl[s]   = rmask;
        bishop_mask_tbl[s] = bmask;

        // enumerate all subsets of the mask
        Bitboard sub = 0;
        do {
            rook_table[s * 4096 + _pext_u64(sub, rmask)] =
                slide_attacks(s, sub, rook_dirs);
            sub = (sub - rmask) & rmask;
        } while (sub);

        sub = 0;
        do {
            bishop_table[s * 512 + _pext_u64(sub, bmask)] =
                slide_attacks(s, sub, bishop_dirs);
            sub = (sub - bmask) & bmask;
        } while (sub);
    }
#endif
}

Bitboard pseudo_attacks(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
        case KNIGHT: return knight_attacks[s];
        case BISHOP: return bishop_attacks(s, occ);
        case ROOK:   return rook_attacks(s, occ);
        case QUEEN:  return bishop_attacks(s, occ) | rook_attacks(s, occ);
        case KING:   return king_attacks[s];
        case PAWN:   return 0;
        default:     return 0;
    }
}

bool self_test(long long& checks, long long& fails) {
    checks = 0;
    fails  = 0;
    auto chk = [&](bool c) { ++checks; if (!c) ++fails; };

    // Independent coordinate-based attack generation.
    auto from_deltas = [](int sq, const int deltas[][2], int n, int maxStep) {
        Bitboard b = 0;
        const int f = FILE_OF(sq), r = RANK_OF(sq);
        for (int i = 0; i < n; ++i) {
            const int df = deltas[i][0], dr = deltas[i][1];
            for (int step = 1; step <= maxStep; ++step) {
                const int cf = f + df * step, cr = r + dr * step;
                if (cf < 0 || cf > 7 || cr < 0 || cr > 7) break;
                b |= Bitboard(1) << SQ(cr, cf);
                if (maxStep == 1) break;
            }
        }
        return b;
    };

    const int knightD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    const int kingD[8][2]   = {{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};
    const int wpawnD[2][2]  = {{-1,1},{1,1}};
    const int bpawnD[2][2]  = {{-1,-1},{1,-1}};
    const int rookD[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};
    const int bishopD[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};

    for (int s = 0; s < SQ_NB; ++s) {
        chk(knight_attacks[s] == from_deltas(s, knightD, 8, 1));
        chk(king_attacks[s]   == from_deltas(s, kingD, 8, 1));
        chk(pawn_attacks[WHITE][s] == from_deltas(s, wpawnD, 2, 1));
        chk(pawn_attacks[BLACK][s] == from_deltas(s, bpawnD, 2, 1));
    }

    // Sliding attacks: compare the lookup table against the ray-based reference
    // over randomized occupancies (including edge squares and bits outside the
    // relevant mask).
    uint64_t rng = 0x123456789ABCDEF0ULL;
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };

    for (int s = 0; s < SQ_NB; ++s) {
        for (int it = 0; it < 3000; ++it) {
            const Bitboard occ = next();
            const Bitboard rook = slide_attacks(s, occ, rookD);
            const Bitboard bishop = slide_attacks(s, occ, bishopD);
            chk(pseudo_attacks(ROOK, Square(s), occ) == rook);
            chk(pseudo_attacks(BISHOP, Square(s), occ) == bishop);
            chk(pseudo_attacks(QUEEN, Square(s), occ) == (rook | bishop));
        }
    }
    return fails == 0;
}

}  // namespace bb