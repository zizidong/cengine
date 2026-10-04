#include "nnue_full.h"

#include "bitboard.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <zlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
inline uint64_t rdtsc_now() { return __rdtsc(); }
#else
inline uint64_t rdtsc_now() { return 0; }
#endif

// Self-contained scalar port of Stockfish's modern NNUE for cengine.
//
// Architecture (must match the .nnue file: version 0x6A448AFA):
//   Features: HalfKAv2_hm (22528) + FullThreats (59808) + PP_3Wide (4560)
//   FeatureTransformer: 1024 int16 per perspective; output activation
//     clamp(acc[j], 0, 255) * clamp(acc[j + 512], 0, 255) / 512
//   Network (per output bucket): fc_0 1024->32, concat
//     [SqrClippedReLU(fc0) | ClippedReLU(fc0) | SqrClippedReLU(fc1) | ClippedReLU(fc1)],
//     fc_1 64->32, fc_2 128->1, plus the skip fc0_out[30] - fc0_out[31].
//   Value from the mover's POV:
//     (psqt[stm] - psqt[nstm]) / 2 / OutputScale + positional / OutputScale
//
// A scalar Stockfish build loads every weight array verbatim
// (AffineTransform::get_weight_index is the identity without SIMD), and this
// port is scalar, so the memory layout is weights[input + output * PaddedIn].
//
// Evaluation is incremental when driven by the searcher (acc_set_root /
// acc_push / acc_pop): each node only applies the symmetric difference of the
// active feature sets to its parent's accumulator.  evaluate() falls back to a
// full refresh whenever the cached state does not match the position.

namespace nnue_full {

// ---- Architecture constants (nnue_architecture.h / nnue_common.h) ----
constexpr int L1              = 1024;
constexpr int L2              = 32;
constexpr int L3              = 32;  // fc_1 outputs
constexpr int PSQTBuckets     = 8;
constexpr int LayerStacks     = 8;
constexpr int WeightScaleBits = 6;
constexpr int FtMaxVal        = 255;
constexpr int OutputScale     = 16;
constexpr int HiddenOneVal    = 128;

constexpr uint32_t kVersion = 0x6A448AFAu;

// ---- Small net (RukChess-style, zlib-compressed) ----
// Format: zlib stream of  version(0x7AF32F20) hash desclen desc
//   ft_hash(u32)  FT-bias[L1] (leb i16)  FT-weights[22528*L1] (leb i16)
//   PSQT[22528*8] (leb i32)  then LayerStacks x8:
//     u32 fc_hash, l1: i32 bias[SL2] + i8 w[SL2*L1]
//                   l2: i32 bias[SL3] + i8 w[SL3*SL2_PAD]   (input padded to 32)
//                   out: i32 bias[1]  + i8 w[1*SL3]
// Plain ClippedReLU between layers (no sqr-concat, no skip), psqt is a
// separate head.  Feature set is HalfKAv2_hm^ -- identical indices to the big
// net's PSQ part, so it reuses halfka_make_index().
constexpr uint32_t kSmallVersion = 0x7AF32F20u;
constexpr int      SL2          = 16;
constexpr int      SL3          = 32;
constexpr int      SL2_PAD      = 32;  // l2 input is padded to 32 by the spec


// Feature-transformer accumulator element type. int16 halves memory traffic and
// doubles AVX2 lane width; int32 is exact but slower. The network must keep the
// accumulated sum within the chosen range (Stockfish uses int16).
using AccT = int16_t;

// ---- Feature dimensions ----
constexpr int PSQ_NB                = 11 * 64;                      // 704
constexpr int PsqDimensions         = 64 * PSQ_NB / 2;              // 22528
constexpr int ThreatDimensions      = 59808;
constexpr int PawnIds               = 96;
constexpr int PairDimensions        = PawnIds * (PawnIds - 1) / 2;  // 4560
constexpr int PairIndexBase         = ThreatDimensions;
constexpr int ThreatAndPpDimensions = ThreatDimensions + PairDimensions;

namespace {

// ---- File reader ----
struct Reader {
    const uint8_t* p;
    size_t         size;
    size_t         pos = 0;
    bool           ok  = true;

    void expect(size_t n) {
        if (pos + n > size) ok = false;
    }

    uint32_t read_u32_le() {
        expect(4);
        if (!ok) return 0;
        uint32_t r = uint32_t(p[pos]) | (uint32_t(p[pos + 1]) << 8)
                   | (uint32_t(p[pos + 2]) << 16) | (uint32_t(p[pos + 3]) << 24);
        pos += 4;
        return r;
    }

    int8_t read_i8() {
        expect(1);
        if (!ok) return 0;
        return int8_t(p[pos++]);
    }

    // Signed LEB128 value (same decoding as Stockfish read_leb_128_detail).
    int32_t read_leb128() {
        uint32_t result = 0;
        int      shift  = 0;
        uint8_t  byte;
        do {
            expect(1);
            if (!ok) return 0;
            byte = p[pos++];
            result |= uint32_t(byte & 0x7f) << (shift % 32);
            shift += 7;
        } while (byte & 0x80);
        return int32_t((shift >= 32 || (byte & 0x40) == 0) ? result
                                                          : result | ~((1u << shift) - 1));
    }

    // Read `count` signed LEB128 values after the COMPRESSED_LEB128 header,
    // verifying exactly the declared number of bytes was consumed.
    template<typename T>
    bool read_leb128_block(size_t count, T* out) {
        static const char kMagic[] = "COMPRESSED_LEB128";
        constexpr int     kMagicLen = int(sizeof(kMagic)) - 1;
        expect(size_t(kMagicLen) + 4);
        if (!ok) return false;
        if (std::memcmp(p + pos, kMagic, kMagicLen) != 0) {
            ok = false;
            return false;
        }
        pos += kMagicLen;
        const uint32_t bytes_left = uint32_t(p[pos]) | (uint32_t(p[pos + 1]) << 8)
                                  | (uint32_t(p[pos + 2]) << 16) | (uint32_t(p[pos + 3]) << 24);
        pos += 4;
        const size_t start = pos;
        for (size_t i = 0; i < count; ++i) out[i] = T(read_leb128());
        if (!ok || pos - start != bytes_left) {
            ok = false;
            return false;
        }
        return true;
    }

    // Read `count` single-byte objects verbatim.
    template<typename T>
    bool read_raw(size_t count, T* out) {
        const size_t bytes = count * sizeof(T);
        expect(bytes);
        if (!ok) return false;
        std::memcpy(out, p + pos, bytes);
        pos += bytes;
        return true;
    }
};

// ---- HalfKAv2_hm feature tables (features/half_ka_v2_hm.h) ----
constexpr int kPS_WPawn = 0, kPS_BPawn = 64,
              kPS_WKnight = 128, kPS_BKnight = 192,
              kPS_WBishop = 256, kPS_BBishop = 320,
              kPS_WRook = 384, kPS_BRook = 448,
              kPS_WQueen = 512, kPS_BQueen = 576,
              kPS_King = 640;

// By Stockfish Piece index (0..15); [perspective 0 = White, 1 = Black].
constexpr int kHalfKA_PieceSquareIndex[2][16] = {
    {0, kPS_WPawn, kPS_WKnight, kPS_WBishop, kPS_WRook, kPS_WQueen, kPS_King, 0,
     0, kPS_BPawn, kPS_BKnight, kPS_BBishop, kPS_BRook, kPS_BQueen, kPS_King, 0},
    {0, kPS_BPawn, kPS_BKnight, kPS_BBishop, kPS_BRook, kPS_BQueen, kPS_King, 0,
     0, kPS_WPawn, kPS_WKnight, kPS_WBishop, kPS_WRook, kPS_WQueen, kPS_King, 0},
};

// KingBuckets (0..31), multiplied by PSQ_NB at use.
constexpr int kHalfKA_KingBuckets[64] = {
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
};

// HalfKAv2_hm orientation: mirror when the king is on files a-d.
constexpr int kHalfKA_OrientTBL[64] = {
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
    7, 7, 7, 7, 0, 0, 0, 0,
};

// FullThreats / PP_3Wide orientation: mirror when the king is on files e-h.
constexpr int kOrientTBL[64] = {
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
    0, 0, 0, 0, 7, 7, 7, 7,
};

// Pawn diagonal threats target knights and rooks only (pawn-pawn relations are
// covered by PP_3Wide).
constexpr int kThreat_numValidTargets[16] = {0, 4, 10, 8, 8, 10, 0, 0,
                                             0, 4, 10, 8, 8, 10, 0, 0};

// [attackerType - 1][attackedType - 1], types 1..6 -> 0..5.
constexpr int kThreat_map[6][6] = {
    {-1,  0, -1,  1, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3, -1, -1},
    { 0,  1,  2,  3,  4, -1},
    {-1, -1, -1, -1, -1, -1},
};

// Stockfish piece order used while building the threat LUTs.
constexpr int kAllPieces[12] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14};

// -- Incremental accumulator sizing --
constexpr int MaxAccPly   = 130;  // MAX_PLY + 2
constexpr int MaxListSize = 640;  // threats (<=256) + pawn pairs (<=256) + margin

struct IndexList {
    int      n = 0;
    uint16_t v[MaxListSize];

    void push(int idx) {
        if (n < MaxListSize) v[n++] = uint16_t(idx);
    }
    void clear() { n = 0; }

    // 2-pass LSD radix sort (uint16 values) for larger lists; insertion sort for
    // the small per-move lists. Much faster than std::sort overall.
    void sort() {
        if (n < 2) return;
        if (n <= 24) {
            for (int i = 1; i < n; ++i) {
                const uint16_t key = v[i];
                int j = i - 1;
                while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
                v[j + 1] = key;
            }
            return;
        }
        uint16_t tmp[MaxListSize];
        uint16_t cnt[256];

        for (int i = 0; i < 256; ++i) cnt[i] = 0;
        for (int i = 0; i < n; ++i) ++cnt[v[i] & 0xFF];
        int sum = 0;
        for (int i = 0; i < 256; ++i) { const int c = cnt[i]; cnt[i] = uint16_t(sum); sum += c; }
        for (int i = 0; i < n; ++i) tmp[cnt[v[i] & 0xFF]++] = v[i];

        for (int i = 0; i < 256; ++i) cnt[i] = 0;
        for (int i = 0; i < n; ++i) ++cnt[tmp[i] >> 8];
        sum = 0;
        for (int i = 0; i < 256; ++i) { const int c = cnt[i]; cnt[i] = uint16_t(sum); sum += c; }
        for (int i = 0; i < n; ++i) v[cnt[tmp[i] >> 8]++] = tmp[i];
    }
};

// ---- cengine Piece -> Stockfish Piece ----
int sf_piece(int cp) {
    if (cp == NO_PIECE || cp < 0 || cp > 13) return 0;
    return cp + 1;
}
int sf_type(int sfp) { return sfp & 7; }    // 1..6
int sf_color(int sfp) { return sfp >> 3; }  // 0..1

int lsb_sq(uint64_t b) { return __builtin_ctzll(b); }

int constexpr_popcount(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return int((x * 0x0101010101010101ULL) >> 56);
}

// ---- HalfKAv2_hm make_index ----
int halfka_make_index(int perspective, int s, int sfp, int ksq) {
    const int flip = 56 * perspective;
    return (s ^ kHalfKA_OrientTBL[ksq] ^ flip) + kHalfKA_PieceSquareIndex[perspective][sfp]
         + kHalfKA_KingBuckets[ksq ^ flip] * PSQ_NB;
}

// ---- FullThreats LUT construction (features/full_threats.cpp) ----
uint64_t pseudo_attacks_empty(int t, int from) {
    switch (t) {
        case 2: return bb::knight_attacks[from];
        case 6: return bb::king_attacks[from];
        case 3: return bb::pseudo_attacks(BISHOP, Square(from), 0);
        case 4: return bb::pseudo_attacks(ROOK, Square(from), 0);
        case 5: return bb::pseudo_attacks(QUEEN, Square(from), 0);
        default: return 0;
    }
}

uint64_t pawn_attacks_bb(int piece, int from) {
    return (piece < 8) ? bb::pawn_attacks[WHITE][from] : bb::pawn_attacks[BLACK][from];
}

struct ThreatLUT {
    std::vector<int> offsets;     // [piece][from]
    std::vector<int> index_lut1;  // [attacker][attacked][from < to]
    std::vector<int> index_lut2;  // [attacker][from][to]
};
ThreatLUT g_threat;

void init_full_threats() {
    // offsets[piece][from]: running offset within the piece's own block.
    // Pawns only accumulate on ranks 2-7 (SQ_A2..SQ_H7).
    std::vector<int> offs(16 * 64, 0);
    std::vector<int> helper(16, 0);
    int cumulativeOffset = 0;
    for (int piece : kAllPieces) {
        const int t = sf_type(piece);
        int cumulativePieceOffset = 0;
        for (int from = 0; from < 64; ++from) {
            offs[piece * 64 + from] = cumulativePieceOffset;
            if (t != 1)
                cumulativePieceOffset += constexpr_popcount(pseudo_attacks_empty(t, from));
            else if (from >= 8 && from <= 55)
                cumulativePieceOffset += constexpr_popcount(pawn_attacks_bb(piece, from));
        }
        helper[piece] = cumulativeOffset;
        cumulativeOffset += kThreat_numValidTargets[piece] * cumulativePieceOffset;
    }
    g_threat.offsets = std::move(offs);

    // index_lut2[piece][from][to] = popcount of attacked squares below `to`.
    std::vector<int> lut2(16 * 64 * 64, 0);
    for (int piece : kAllPieces) {
        const int t = sf_type(piece);
        for (int from = 0; from < 64; ++from) {
            const uint64_t attacks =
              (t != 1) ? pseudo_attacks_empty(t, from) : pawn_attacks_bb(piece, from);
            for (int to = 0; to < 64; ++to)
                lut2[piece * 64 * 64 + from * 64 + to] =
                  constexpr_popcount(((uint64_t(1) << to) - 1) & attacks);
        }
    }
    g_threat.index_lut2 = std::move(lut2);

    // index_lut1[attacker][attacked][from < to]: base feature offset, or
    // ThreatDimensions when the relation is excluded.
    std::vector<int> lut1(16 * 16 * 2, ThreatDimensions);
    for (int attacker : kAllPieces) {
        const int at = sf_type(attacker);
        int64_t   cumulativePieceOffset = 0;
        for (int from = 0; from < 64; ++from) {
            uint64_t a = (at != 1) ? pseudo_attacks_empty(at, from)
                                   : pawn_attacks_bb(attacker, from);
            if (at == 1 && (from < 8 || from > 55)) a = 0;
            cumulativePieceOffset += constexpr_popcount(a);
        }
        for (int attacked : kAllPieces) {
            const bool    enemy = (attacker ^ attacked) == 8;
            const int     dt    = sf_type(attacked);
            const int     m     = kThreat_map[at - 1][dt - 1];
            const bool    semi  = (at == dt) && (enemy || at != 1);
            const int64_t feature =
              helper[attacker]
              + (sf_color(attacked) * (kThreat_numValidTargets[attacker] / 2) + m)
                  * cumulativePieceOffset;
            const bool excluded = m < 0;
            lut1[(attacker * 16 + attacked) * 2 + 0] = excluded ? ThreatDimensions : int(feature);
            lut1[(attacker * 16 + attacked) * 2 + 1] =
              (excluded || semi) ? ThreatDimensions : int(feature);
        }
    }
    g_threat.index_lut1 = std::move(lut1);
}

// ---- FullThreats active features (features/full_threats.cpp) ----

// A lightweight position view used both for full feature enumeration and for
// the incremental per-piece delta.
struct View {
    Bitboard occ            = 0;
    Bitboard bt[PIECE_TYPE_NB] = {};
    Bitboard bc[COLOR_NB]   = {};
    Piece    pcs[SQ_NB]     = {};
};

View make_view(const Board& b) {
    View v;
    v.occ = b.occupancy();
    for (int i = 0; i < PIECE_TYPE_NB; ++i) v.bt[i] = b.by_type[i];
    for (int i = 0; i < COLOR_NB; ++i) v.bc[i] = b.by_color[i];
    std::memcpy(v.pcs, b.board, sizeof(v.pcs));
    return v;
}

Bitboard attackers_masks(Square s, Bitboard occ, const Bitboard bt[PIECE_TYPE_NB],
                         const Bitboard bc[COLOR_NB]) {
    return (bt[PAWN] & bc[WHITE] & bb::pawn_attacks[BLACK][s])
         | (bt[PAWN] & bc[BLACK] & bb::pawn_attacks[WHITE][s])
         | (bt[KNIGHT] & bb::knight_attacks[s])
         | (bt[BISHOP] & bb::pseudo_attacks(BISHOP, s, occ))
         | (bt[ROOK]   & bb::pseudo_attacks(ROOK, s, occ))
         | (bt[QUEEN]  & bb::pseudo_attacks(QUEEN, s, occ))
         | (bt[KING]   & bb::king_attacks[s]);
}

// Compute the attacker code and the attack set (already intersected with the
// valid threat targets) of the piece at square `a` in `v`.
bool attacker_attacks(const View& v, int a, int& attacker, Bitboard& attacks) {
    const Piece ap = v.pcs[a];
    if (ap == NO_PIECE) return false;
    if (type_of(ap) == PAWN) {
        const int c = color_of(ap);
        attacker = (c == WHITE) ? 1 : 9;
        attacks  = bb::pawn_attacks[c][a] & (v.bt[KNIGHT] | v.bt[ROOK]);
    } else {
        const int      pt    = type_of(ap);
        attacker             = (color_of(ap) == WHITE) ? pt + 1 : pt + 1 + 8;
        const Bitboard minor = v.bt[PAWN] | v.bt[KNIGHT] | v.bt[BISHOP] | v.bt[ROOK];
        const Bitboard targets = (pt == KNIGHT || pt == QUEEN) ? (minor | v.bt[QUEEN]) : minor;
        attacks = bb::pseudo_attacks(PieceType(pt), Square(a), v.occ) & targets;
    }
    return true;
}

inline int threat_index_for(int orientation, int swap, int attacker, int from, int to, int attacked) {
    const int attacker_o = attacker ^ swap;
    const int from_o     = from ^ orientation;
    const int to_o       = to ^ orientation;
    const int attacked_o = attacked ^ swap;
    const int lut1 =
      g_threat.index_lut1[(attacker_o * 16 + attacked_o) * 2 + (from_o < to_o ? 1 : 0)];
    if (lut1 >= ThreatDimensions) return -1;
    return lut1 + g_threat.offsets[attacker_o * 64 + from_o]
         + g_threat.index_lut2[attacker_o * 64 * 64 + from_o * 64 + to_o];
}

// Emit the full threat feature set of the attacker at `from`.
void threat_emit_one(const View& v, int orientation, int swap, int from, IndexList& out) {
    int      attacker;
    Bitboard attacks;
    if (!attacker_attacks(v, from, attacker, attacks)) return;
    while (attacks) {
        const int to     = lsb_sq(attacks);
        attacks &= attacks - 1;
        const int idx = threat_index_for(orientation, swap, attacker, from, to,
                                         sf_piece(v.pcs[to]));
        if (idx >= 0) out.push(idx);
    }
}

void threat_append_active(const Board& b, int perspective, IndexList& out) {
    const View v = make_view(b);
    const int  ksq         = b.king_square(Color(perspective));
    const int  orientation = kOrientTBL[ksq] ^ (56 * perspective);
    const int  swap        = 8 * perspective;
    for (int s = 0; s < SQ_NB; ++s)
        threat_emit_one(v, orientation, swap, s, out);
}

// ---- PP_3Wide features (features/pp_3wide.cpp) ----
uint64_t pawn_pair_bb(int s) {
    const int      f     = FILE_OF(s);
    const uint64_t file  = uint64_t(0x0101010101010101ULL) << f;
    uint64_t       files = file;
    if (f > 0) files |= file >> 1;
    if (f < 7) files |= file << 1;
    files &= ~0xFFULL;                // no rank 1
    files &= ~0xFF00000000000000ULL;  // no rank 8
    files &= ~(uint64_t(1) << s);     // not the pawn itself
    return files;
}

int make_pawn_id(int color, int square) { return 48 * color + square - 8; }  // rank 2..7

// Emit the pawn-pair features owned by the pawn at `from` (owner = lower square
// index for same-colour pairs, the white pawn for mixed pairs).
void pair_emit_one(const View& v, int perspective, int orientation, int from, IndexList& out) {
    const Piece ap = v.pcs[from];
    if (ap == NO_PIECE || type_of(ap) != PAWN) return;

    const int      c     = color_of(ap);
    const Bitboard white = v.bt[PAWN] & v.bc[WHITE];
    const Bitboard black = v.bt[PAWN] & v.bc[BLACK];
    const Bitboard band  = pawn_pair_bb(from);

    auto emitPair = [&](int color, int f, int pairedColor, int t) {
        const int from_o   = f ^ orientation;
        const int to_o     = t ^ orientation;
        const int color_o  = color ^ perspective;
        const int paired_o = pairedColor ^ perspective;
        const int idA      = make_pawn_id(color_o, from_o);
        const int idB      = make_pawn_id(paired_o, to_o);
        const int hi       = std::max(idA, idB);
        const int lo       = std::min(idA, idB);
        const int idx      = hi * (hi - 1) / 2 + lo + PairIndexBase;
        if (idx < ThreatAndPpDimensions) out.push(idx);
    };

    if (c == WHITE) {
        for (Bitboard x = band & white; x;) {
            const int to = lsb_sq(x); x &= x - 1;
            if (to > from) emitPair(WHITE, from, WHITE, to);
        }
        for (Bitboard x = band & black; x;) {
            const int to = lsb_sq(x); x &= x - 1;
            emitPair(WHITE, from, BLACK, to);
        }
    } else {
        for (Bitboard x = band & black; x;) {
            const int to = lsb_sq(x); x &= x - 1;
            if (to > from) emitPair(BLACK, from, BLACK, to);
        }
    }
}

void pair_append_active(const Board& b, int perspective, IndexList& out) {
    const View v = make_view(b);
    const int  ksq         = b.king_square(Color(perspective));
    const int  orientation = kOrientTBL[ksq] ^ (56 * perspective);
    for (int s = 0; s < SQ_NB; ++s)
        pair_emit_one(v, perspective, orientation, s, out);
}

// Fused threat+pair enumeration: one board view and one 64-square pass.
void append_active_all(const Board& b, int perspective, IndexList& out) {
    const View v = make_view(b);
    const int  ksq         = b.king_square(Color(perspective));
    const int  orientation = kOrientTBL[ksq] ^ (56 * perspective);
    const int  swap        = 8 * perspective;
    for (int s = 0; s < SQ_NB; ++s) {
        threat_emit_one(v, orientation, swap, s, out);
        pair_emit_one(v, perspective, orientation, s, out);
    }
}

// ---- Weights ----
struct Net {
    std::vector<int16_t> biases;         // L1
std::vector<int8_t>  threatWeights;  // ThreatAndPpDimensions * L1
    std::vector<int32_t> threatPsqt;     // ThreatAndPpDimensions * PSQTBuckets
    std::vector<int16_t> weights;        // PsqDimensions * L1
    std::vector<int32_t> psqtWeights;    // PSQTBuckets * PsqDimensions

    std::vector<std::vector<int32_t>> fc0_bias;  // [LayerStacks][L2]
    std::vector<std::vector<int8_t>>  fc0_w;     // [LayerStacks][L1 * L2]
    std::vector<std::vector<int32_t>> fc1_bias;  // [LayerStacks][L3]
    std::vector<std::vector<int8_t>>  fc1_w;     // [LayerStacks][2 * L2 * L3]
    std::vector<std::vector<int32_t>> fc2_bias;  // [LayerStacks][1]
    std::vector<std::vector<int8_t>>  fc2_w;     // [LayerStacks][2 * L2 + 2 * L3]
};

Net         g_net;
bool        g_loaded = false;
std::string g_path;

// Incremental FullThreats/PP accumulator (A/B toggle: NNUE_LEGACY_BIGNET).
bool g_threat_inc = true;

// Small (cache-resident) net: modern HalfKAv2_hm FT + old-style 16/32/1 stack.
struct SmallNet {
    std::vector<int16_t> biases;       // L1
    std::vector<int16_t> weights;      // PsqDimensions * L1
    std::vector<int32_t> psqtWeights;  // PSQTBuckets * PsqDimensions

    std::vector<std::vector<int32_t>> fc0_bias;  // [LayerStacks][SL2]
    std::vector<std::vector<int8_t>>  fc0_w;     // [LayerStacks][SL2 * L1]
    std::vector<std::vector<int32_t>> fc1_bias;  // [LayerStacks][SL3]
    std::vector<std::vector<int8_t>>  fc1_w;     // [LayerStacks][SL3 * SL2_PAD]
    std::vector<std::vector<int32_t>> fc2_bias;  // [LayerStacks][1]
    std::vector<std::vector<int8_t>>  fc2_w;     // [LayerStacks][1 * SL3]
};

SmallNet    g_snet;
bool        g_sloaded = false;
std::string g_spath;
int         g_sL1 = 128;  // small net feature-transformer width (per perspective)
int         g_bignet_ply = 2;   // big net used for ply <= this
int         g_bignet_pv  = 1;   // also use the big net at PV nodes

// ---- Incremental accumulator state ----
// Only the small net is maintained incrementally (cheap: HalfKAv2_hm PSQ rows).
// The big net is evaluated with a full refresh, but only on the few big nodes
// (PV / shallow ply), so its expensive threat/pair features are computed on
// demand instead of on every move.
struct AccEntry {
    alignas(32) AccT    accS[2][L1];     // small net FT (PSQ only)
    int32_t             psqtS[2][PSQTBuckets];
    alignas(32) AccT    acc[2][L1];      // big net FT (PSQ + threats/pairs)
    int32_t             psqt[2][PSQTBuckets];
    IndexList           thr[2];          // active threat+pair features (per persp)
    Key       key      = 0;
    bool      valid    = false;
    bool      big      = false;
    bool      bigValid = false;  // big accumulators valid for this node
};

struct AccStack {
    std::vector<AccEntry> entries;
    int                   size = 0;
};
thread_local AccStack g_acc;

// Instrumentation (relaxed atomics; negligible vs an evaluation).
std::atomic<uint64_t> g_inc_evals{0};
std::atomic<uint64_t> g_full_evals{0};
std::atomic<uint64_t> g_push_count{0};
std::atomic<uint64_t> g_diff_applies{0};
std::atomic<uint64_t> g_big_evals{0};

// Cycle counters for the NNUE hot path (single-thread profiling).
uint64_t g_cyc_build  = 0;  // build_features (psq + threat feature lists)
uint64_t g_cyc_apply  = 0;  // apply_diff (incremental weight updates)
uint64_t g_cyc_finish = 0;  // transform + layer stacks
uint64_t g_cyc_build_append = 0;  // feature enumeration
uint64_t g_cyc_build_sort   = 0;  // sorting the feature lists

// Add/subtract one weight row of `len` int16 lanes into an int16 accumulator.
template<typename W>
inline void apply_weight_row_len(AccT* acc, const W* w, int sign, int len) {
    int j = 0;
#if defined(__AVX2__)
    if constexpr (sizeof(AccT) == 2) {
        for (; j + 16 <= len; j += 16) {
            __m256i v;
            if constexpr (sizeof(W) == 2) {
                v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j));
            } else {
                v = _mm256_cvtepi8_epi16(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
            }
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + j));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j),
                                sign > 0 ? _mm256_add_epi16(a, v) : _mm256_sub_epi16(a, v));
        }
    }
#endif
    if (sign > 0)
        for (; j < len; ++j) acc[j] = AccT(acc[j] + int32_t(w[j]));
    else
        for (; j < len; ++j) acc[j] = AccT(acc[j] - int32_t(w[j]));
}

// Apply one feature row of `len` lanes (used by the small net, whose FT is
// narrower than L1) plus its psqt row.
template<typename W>
inline void apply_feature_len(const W* wbase, const int32_t* pbase, int idx, int sign, AccT* acc,
                              int32_t* pst, int len) {
    apply_weight_row_len(acc, wbase + size_t(idx) * size_t(len), sign, len);
    const int32_t* pw = pbase + size_t(idx) * PSQTBuckets;
    for (int k = 0; k < PSQTBuckets; ++k) pst[k] += sign * pw[k];
}

// Add (sign > 0) or subtract (sign < 0) one int8/int16 weight row into an int16
// accumulator of length L1. 16-wide with AVX2 (int16 lanes).
template<typename W>
inline void apply_weight_row(AccT* acc, const W* w, int sign) {
    int j = 0;
#if defined(__AVX2__)
    if constexpr (sizeof(AccT) == 2) {
        if (sign > 0) {
            for (; j + 32 <= L1; j += 32) {
                __m256i w0, w1;
                if constexpr (sizeof(W) == 2) {
                    w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j));
                    w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j + 16));
                } else {
                    w0 = _mm256_cvtepi8_epi16(
                      _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
                    w1 = _mm256_cvtepi8_epi16(
                      _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j + 16)));
                }
                __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j));
                __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j), _mm256_add_epi16(a0, w0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j + 16),
                                   _mm256_add_epi16(a1, w1));
            }
        } else {
            for (; j + 32 <= L1; j += 32) {
                __m256i w0, w1;
                if constexpr (sizeof(W) == 2) {
                    w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j));
                    w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j + 16));
                } else {
                    w0 = _mm256_cvtepi8_epi16(
                      _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
                    w1 = _mm256_cvtepi8_epi16(
                      _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j + 16)));
                }
                __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j));
                __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j + 16));
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j), _mm256_sub_epi16(a0, w0));
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j + 16),
                                   _mm256_sub_epi16(a1, w1));
            }
        }
        for (; j + 16 <= L1; j += 16) {
            __m256i wv;
            if constexpr (sizeof(W) == 2)
                wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + j));
            else
                wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
            __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + j));
            __m256i r = (sign > 0) ? _mm256_add_epi16(a, wv) : _mm256_sub_epi16(a, wv);
            _mm256_store_si256(reinterpret_cast<__m256i*>(acc + j), r);
        }
    } else {
        for (; j + 8 <= L1; j += 8) {
            __m256i wv;
            if constexpr (sizeof(W) == 2)
                wv = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)));
            else
                wv = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + j)));
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + j));
            __m256i r = (sign > 0) ? _mm256_add_epi32(a, wv) : _mm256_sub_epi32(a, wv);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + j), r);
        }
    }
#endif
    for (; j < L1; ++j) acc[j] = AccT(acc[j] + sign * int(w[j]));
}

// Apply a single feature's weight row (+1 add / -1 remove) to an accumulator.
template<typename W>
inline void apply_feature(const W* wbase, const int32_t* pbase, int idx, int sign, AccT* acc,
                          int32_t* pst) {
    apply_weight_row<W>(acc, wbase + size_t(idx) * L1, sign);
    const int32_t* pw = pbase + size_t(idx) * PSQTBuckets;
    for (int k = 0; k < PSQTBuckets; ++k) pst[k] += sign * pw[k];
    ++g_diff_applies;
}

// Merges two already-sorted feature index lists and applies the symmetric
// (multiset) difference of `parent` -> `child` to the accumulator.
// Lists must be sorted (see build_features).
template<typename W>
void apply_diff(const W* wbase, const int32_t* pbase, const IndexList& parent,
                const IndexList& child, AccT* acc, int32_t* pst) {
    const uint16_t* pa = parent.v;
    const uint16_t* ch = child.v;
    const int       np = parent.n;
    const int       nc = child.n;

    auto apply = [&](int sign, uint16_t idx) {
        ++g_diff_applies;
        const W*       w  = wbase + size_t(idx) * L1;
        const int32_t* pw = pbase + size_t(idx) * PSQTBuckets;
        apply_weight_row<W>(acc, w, sign);
        for (int k = 0; k < PSQTBuckets; ++k) pst[k] += sign * pw[k];
    };

    int i = 0, j = 0;
    while (i < np && j < nc) {
        if (pa[i] == ch[j]) {  // present in both: cancel
            ++i;
            ++j;
        } else if (pa[i] < ch[j]) {
            apply(-1, pa[i++]);  // removed
        } else {
            apply(+1, ch[j++]);  // added
        }
    }
    for (; i < np; ++i) apply(-1, pa[i]);
    for (; j < nc; ++j) apply(+1, ch[j]);
}

void build_features(const Board& b, int perspective, IndexList& psqOut, IndexList& thrOut) {
    psqOut.clear();
    thrOut.clear();
    const uint64_t ta0 = rdtsc_now();
    const int ksq = b.king_square(Color(perspective));
    uint64_t  occ = b.occupancy();
    while (occ) {
        const int s = lsb_sq(occ);
        occ &= occ - 1;
        const int sfp = sf_piece(b.piece_on(Square(s)));
        if (sfp == 0) continue;
        psqOut.push(halfka_make_index(perspective, s, sfp, ksq));
    }
    threat_append_active(b, perspective, thrOut);
    pair_append_active(b, perspective, thrOut);
    const uint64_t ta1 = rdtsc_now();
    psqOut.sort();
    thrOut.sort();
    g_cyc_build_append += ta1 - ta0;
    g_cyc_build_sort   += rdtsc_now() - ta1;
}

// ---- Scalar propagation helpers ----
// Dense affine transform in Stockfish's scalar layout:
//   output[o] = bias[o] + sum_i weights[o * inDim + i] * input[i]
// input is uint8 (0..127), weights are int8. Vectorized with AVX2.
void affine(const int32_t* bias, const int8_t* w, int inDim, int outDim, const uint8_t* input,
            int32_t* output) {
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
    for (int o = 0; o < outDim; ++o) {
        const int8_t* row = w + size_t(o) * size_t(inDim);
        __m256i sum = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= inDim; i += 32) {
            __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
            __m256i wt = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
            __m256i p16 = _mm256_maddubs_epi16(in, wt);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p16, ones));
        }
        alignas(32) int32_t tmp[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), sum);
        int32_t s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < inDim; ++i) s += int(input[i]) * int(row[i]);
        output[o] = bias[o] + s;
    }
#else
    for (int o = 0; o < outDim; ++o) output[o] = bias[o];
    for (int i = 0; i < inDim; ++i) {
        if (input[i]) {
            const int8_t* row = w + i;
            const int     v   = input[i];
            for (int o = 0; o < outDim; ++o) output[o] += row[o * inDim] * v;
        }
    }
#endif
}

// ClippedReLU: clamp(x >> shift, 0, 127).
void clipped_relu(const int32_t* input, int dim, int shift, uint8_t* output) {
    for (int i = 0; i < dim; ++i)
        output[i] = uint8_t(std::clamp(input[i] >> shift, 0, 127));
}

// SqrClippedReLU: min(127, (x * x) >> (2 * shift + 7)).
void sqr_clipped_relu(const int32_t* input, int dim, int shift, uint8_t* output) {
    for (int i = 0; i < dim; ++i)
        output[i] = uint8_t(
          std::min<long long>(127, ((long long)(input[i]) * input[i]) >> (2 * shift + 7)));
}

// One output-bucket layer stack (nnue_architecture.h NetworkArchitecture).
int propagate_stack(const Net& n, int st, const uint8_t* transformed) {
    int32_t fc0_out[L2];
    affine(n.fc0_bias[st].data(), n.fc0_w[st].data(), L1, L2, transformed, fc0_out);

    uint8_t concat[2 * L2 + 2 * L3];
    sqr_clipped_relu(fc0_out, L2, WeightScaleBits + 1, concat);
    clipped_relu(fc0_out, L2, WeightScaleBits + 1, concat + L2);

    int32_t fc1_out[L3];
    affine(n.fc1_bias[st].data(), n.fc1_w[st].data(), 2 * L2, L3, concat, fc1_out);

    sqr_clipped_relu(fc1_out, L3, WeightScaleBits, concat + 2 * L2);
    clipped_relu(fc1_out, L3, WeightScaleBits, concat + 2 * L2 + L3);

    int32_t fc2_out[1];
    affine(n.fc2_bias[st].data(), n.fc2_w[st].data(), 2 * L2 + 2 * L3, 1, concat, fc2_out);

    int32_t fwdOut = fc2_out[0] + fc0_out[L2 - 2] - fc0_out[L2 - 1];

    constexpr int64_t multiplier  = 600 * OutputScale;
    constexpr int64_t denominator = int64_t(HiddenOneVal) * (int64_t(1) << WeightScaleBits) * 2;
    return int((int64_t(fwdOut) * multiplier) / denominator);
}

// FeatureTransformer output for one perspective slot (0 = stm, 1 = opponent):
// 512 bytes each, pair activation.
void transform(int slot, const AccT* acc_in, uint8_t* out) {
    const int offset = (L1 / 2) * slot;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i maxv = _mm256_set1_epi16(FtMaxVal);
    for (int j = 0; j < L1 / 2; j += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_in + j));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_in + j + L1 / 2));
        a = _mm256_min_epi16(_mm256_max_epi16(a, zero), maxv);
        b = _mm256_min_epi16(_mm256_max_epi16(b, zero), maxv);
        // (a * 128 * b) >> 16 == (a * b) / 512  (a,b in 0..255)
        const __m256i p  = _mm256_mulhi_epi16(_mm256_slli_epi16(a, 7), b);
        const __m128i lo = _mm256_castsi256_si128(p);
        const __m128i hi = _mm256_extracti128_si256(p, 1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + offset + j),
                         _mm_packus_epi16(lo, hi));
    }
#else
    for (int j = 0; j < L1 / 2; ++j) {
        const int sum0 = std::clamp(int(acc_in[j]), 0, FtMaxVal);
        const int sum1 = std::clamp(int(acc_in[j + L1 / 2]), 0, FtMaxVal);
        out[offset + j] = uint8_t((unsigned(sum0) * unsigned(sum1)) / 512);
    }
#endif
}

// Small net FeatureTransformer output (g_sL1 wide, pair activation).
void transform_small(const AccT* acc_in, uint8_t* out) {
    const int half = g_sL1 / 2;
    for (int j = 0; j < half; ++j) {
        const int s0 = std::clamp(int(acc_in[j]), 0, FtMaxVal);
        const int s1 = std::clamp(int(acc_in[j + half]), 0, FtMaxVal);
        out[j] = uint8_t((unsigned(s0) * unsigned(s1)) / 512);
    }
}

// Full (non-incremental) big-net accumulator for one position.
struct FullScratch {
    AccT      acc[2][L1];
    int32_t   psqt[2][PSQTBuckets];
    IndexList psq[2];
    IndexList thr[2];
};

void full_entry(const Net& n, const Board& b, FullScratch& e) {
    IndexList empty;
    for (int c = 0; c < 2; ++c) {
        build_features(b, c, e.psq[c], e.thr[c]);
        for (int i = 0; i < L1; ++i) e.acc[c][i] = n.biases[i];
        for (int k = 0; k < PSQTBuckets; ++k) e.psqt[c][k] = 0;
        apply_diff(n.weights.data(), n.psqtWeights.data(), empty, e.psq[c], e.acc[c],
                   e.psqt[c]);
        apply_diff(n.threatWeights.data(), n.threatPsqt.data(), empty, e.thr[c], e.acc[c],
                   e.psqt[c]);
    }
}

// Transformed features + network from an accumulator (the "finish" step).
int finish(const Net& n, int stm, int cnt, const AccT acc[2][L1],
           const int32_t psqt[2][PSQTBuckets]) {
    const int nstm   = stm ^ 1;  // NOT ~stm: stm is a plain int here
    const int bucket = std::clamp((cnt - 1) / 4, 0, PSQTBuckets - 1);
    uint8_t transformed[L1];
    transform(0, acc[stm], transformed);
    transform(1, acc[nstm], transformed);
    const int psqt_val = ((psqt[stm][bucket] - psqt[nstm][bucket]) / 2) / OutputScale;
    const int layer_val = propagate_stack(n, bucket, transformed) / OutputScale;
    return psqt_val + layer_val;
}

// Small net: modern pair-activation FT (1024) -> 16 -> 32 -> 1, plain
// ClippedReLU between layers, no skip, psqt is a separate head.
int finish_small(const SmallNet& n, int stm, int cnt, const AccT acc[2][L1],
                 const int32_t psqt[2][PSQTBuckets]) {
    const int nstm   = stm ^ 1;
    const int bucket = std::clamp((cnt - 1) / 4, 0, PSQTBuckets - 1);

    uint8_t transformed[L1];
    transform_small(acc[stm], transformed);
    transform_small(acc[nstm], transformed + g_sL1 / 2);

    int32_t l1c[SL2];
    affine(n.fc0_bias[bucket].data(), n.fc0_w[bucket].data(), g_sL1, SL2, transformed, l1c);

    // SF-style concat: [SqrClippedReLU(l1) | ClippedReLU(l1)]  (2 * SL2)
    uint8_t l1y[2 * SL2];
    sqr_clipped_relu(l1c, SL2, WeightScaleBits + 1, l1y);
    clipped_relu(l1c, SL2, WeightScaleBits + 1, l1y + SL2);

    int32_t l2c[SL3];
    affine(n.fc1_bias[bucket].data(), n.fc1_w[bucket].data(), 2 * SL2, SL3, l1y, l2c);

    uint8_t l2y[SL3];
    for (int i = 0; i < SL3; ++i)
        l2y[i] = uint8_t(std::clamp(l2c[i] >> WeightScaleBits, 0, 127));

    int32_t fwdOut = n.fc2_bias[bucket][0] + l1c[SL2 - 2] - l1c[SL2 - 1];  // skip
    for (int i = 0; i < SL3; ++i) fwdOut += int32_t(n.fc2_w[bucket][i]) * int32_t(l2y[i]);

    constexpr int64_t multiplier  = 600 * OutputScale;
    constexpr int64_t denominator = int64_t(HiddenOneVal) * (int64_t(1) << WeightScaleBits);
    const int layer_val = int((int64_t(fwdOut) * multiplier) / denominator) / OutputScale;
    const int psqt_val  = ((psqt[stm][bucket] - psqt[nstm][bucket]) / 2) / OutputScale;
    static const bool dbg_small = std::getenv("NNUE_DEBUG_SMALL") != nullptr;
    if (dbg_small)
        std::fprintf(stderr,
                     "[small] bucket=%d l1c0=%d l1c1=%d l2c0=%d l2c1=%d fwdOut=%d layer=%d psqt=%d "
                     "psqtraw=%d,%d fc2bias=%d l2yn0=%d\n",
                     bucket, l1c[0], l1c[1], l2c[0], l2c[1], fwdOut, layer_val, psqt_val,
                     psqt[stm][bucket], psqt[nstm][bucket], n.fc2_bias[bucket][0],
                     int(l2y[0]));
    return psqt_val + layer_val;
}

// Full (non-incremental) small-net evaluation; used only if the accumulator
// stack is unavailable (e.g. the big net is not loaded).
int full_small(const Board& b) {
    static thread_local AccT    aS[2][L1];
    static thread_local int32_t pS[2][PSQTBuckets];
    for (int persp = 0; persp < 2; ++persp) {
        const int ksq = int(b.king_square(Color(persp)));
        for (int i = 0; i < g_sL1; ++i) aS[persp][i] = g_snet.biases[i];
        for (int k = 0; k < PSQTBuckets; ++k) pS[persp][k] = 0;
        uint64_t occ = b.occupancy();
        while (occ) {
            const int s = lsb_sq(occ);
            occ &= occ - 1;
            const int sfp = sf_piece(b.piece_on(Square(s)));
            if (!sfp) continue;
            apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                              halfka_make_index(persp, s, sfp, ksq), +1, aS[persp], pS[persp],
                              g_sL1);
        }
    }
    return finish_small(g_snet, int(b.side_to_move()), constexpr_popcount(b.occupancy()), aS,
                        pS);
}

int evaluate_full(const Board& b) {
    thread_local FullScratch scratch;
    full_entry(g_net, b, scratch);
    return finish(g_net, int(b.side_to_move()), constexpr_popcount(b.occupancy()), scratch.acc,
                  scratch.psqt);
}

}  // namespace

bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    std::fseek(f, 0, SEEK_END);
    const long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 0) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> buf(static_cast<std::size_t>(fsize));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    init_full_threats();

    Reader r{buf.data(), buf.size()};

    const uint32_t version  = r.read_u32_le();
    r.read_u32_le();  // architecture hash
    const uint32_t desc_len = r.read_u32_le();
    if (!r.ok || version != kVersion) return false;
    if (desc_len > 0) {
        r.expect(desc_len);
        r.pos += desc_len;
    }
    if (!r.ok) return false;

    r.read_u32_le();  // FeatureTransformer hash

    Net n;
    n.biases.resize(L1);
    n.threatWeights.resize(size_t(ThreatAndPpDimensions) * L1);
    n.threatPsqt.resize(size_t(ThreatAndPpDimensions) * PSQTBuckets);
    n.weights.resize(size_t(PsqDimensions) * L1);
    n.psqtWeights.resize(size_t(PSQTBuckets) * PsqDimensions);

    // FeatureTransformer: biases, threat weights, threat psqt, pair weights,
    // pair psqt, psq weights, psq psqt (Stockfish read_parameters order).
    if (!r.read_leb128_block(n.biases.size(), n.biases.data())) return false;
    if (!r.read_raw(size_t(ThreatDimensions) * L1, n.threatWeights.data())) return false;
    if (!r.read_leb128_block(size_t(ThreatDimensions) * PSQTBuckets, n.threatPsqt.data()))
        return false;
    if (!r.read_raw(size_t(PairDimensions) * L1,
                    n.threatWeights.data() + size_t(ThreatDimensions) * L1))
        return false;
    if (!r.read_leb128_block(size_t(PairDimensions) * PSQTBuckets,
                             n.threatPsqt.data() + size_t(ThreatDimensions) * PSQTBuckets))
        return false;
    if (!r.read_leb128_block(n.weights.size(), n.weights.data())) return false;
    if (!r.read_leb128_block(n.psqtWeights.size(), n.psqtWeights.data())) return false;

    n.fc0_bias.resize(LayerStacks);
    n.fc0_w.resize(LayerStacks);
    n.fc1_bias.resize(LayerStacks);
    n.fc1_w.resize(LayerStacks);
    n.fc2_bias.resize(LayerStacks);
    n.fc2_w.resize(LayerStacks);

    for (int st = 0; st < LayerStacks; ++st) {
        r.read_u32_le();  // NetworkArchitecture hash

        n.fc0_bias[st].resize(L2);
        for (int i = 0; i < L2; ++i) n.fc0_bias[st][i] = int32_t(r.read_u32_le());
        n.fc0_w[st].resize(size_t(L1) * L2);
        if (!r.read_raw(n.fc0_w[st].size(), n.fc0_w[st].data())) return false;

        n.fc1_bias[st].resize(L3);
        for (int i = 0; i < L3; ++i) n.fc1_bias[st][i] = int32_t(r.read_u32_le());
        n.fc1_w[st].resize(size_t(2 * L2) * L3);
        if (!r.read_raw(n.fc1_w[st].size(), n.fc1_w[st].data())) return false;

        n.fc2_bias[st].resize(1);
        n.fc2_bias[st][0] = int32_t(r.read_u32_le());
        n.fc2_w[st].resize(size_t(2 * L2 + 2 * L3));
        if (!r.read_raw(n.fc2_w[st].size(), n.fc2_w[st].data())) return false;
    }

    if (!r.ok || r.pos != r.size) return false;

    g_net    = std::move(n);
    g_loaded = true;
    g_path   = path;
    g_acc.size = 0;
    return true;
}

// Small (RukChess-style) net: zlib stream, modern HalfKAv2_hm FT + 16/32/1 stack.
bool load_small(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 8) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> comp(static_cast<std::size_t>(fsize));
    if (std::fread(comp.data(), 1, comp.size(), f) != comp.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    std::vector<uint8_t> buf;
    if (comp.size() >= 2 && comp[0] == 0x78 && (comp[1] == 0x9C || comp[1] == 0x01)) {
    buf.resize(1u << 22);
    z_stream zs{};
    if (inflateInit(&zs) != Z_OK) return false;
    zs.next_in  = comp.data();
    zs.avail_in = uInt(comp.size());
    int ret = Z_OK;
    do {
        if (zs.total_out == buf.size()) buf.resize(buf.size() * 2);
        zs.next_out  = buf.data() + zs.total_out;
        zs.avail_out = uInt(buf.size() - zs.total_out);
        ret          = inflate(&zs, Z_NO_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            inflateEnd(&zs);
            
            return false;
        }
    } while (ret != Z_STREAM_END);
    const size_t out_size = zs.total_out;
    inflateEnd(&zs);
    buf.resize(out_size);
    if (out_size < 64) return false;
    } else {
        buf = std::move(comp);  // plain (uncompressed) net file
    }

    Reader r{buf.data(), buf.size()};

    const uint32_t version  = r.read_u32_le();
    r.read_u32_le();  // network hash
    const uint32_t desc_len = r.read_u32_le();
    if (!r.ok || version != kSmallVersion) return false;
    if (desc_len > 0) {
        r.expect(desc_len);
        r.pos += desc_len;
    }
    if (!r.ok) return false;
    const uint32_t ft_hash = r.read_u32_le();  // feature-transformer hash
    // ft_hash == HalfKAv2_hm hash ^ (2 * L1)  -> recover the FT width.
    {
        const uint32_t x = ft_hash ^ 0x7f234cb8u;
        if (x == 0 || (x & 1)) return false;
        const int l1 = int(x / 2);
        if (l1 < 16 || l1 > L1) return false;
        g_sL1 = l1;
    }

    SmallNet n;
    n.biases.resize(g_sL1);
    n.weights.resize(size_t(PsqDimensions) * g_sL1);
    n.psqtWeights.resize(size_t(PSQTBuckets) * PsqDimensions);

    if (!r.read_leb128_block(n.biases.size(), n.biases.data())) return false;
    if (!r.read_leb128_block(n.weights.size(), n.weights.data())) return false;
    if (!r.read_leb128_block(n.psqtWeights.size(), n.psqtWeights.data())) return false;

    n.fc0_bias.resize(LayerStacks);
    n.fc0_w.resize(LayerStacks);
    n.fc1_bias.resize(LayerStacks);
    n.fc1_w.resize(LayerStacks);
    n.fc2_bias.resize(LayerStacks);
    n.fc2_w.resize(LayerStacks);

    for (int st = 0; st < LayerStacks; ++st) {
        r.read_u32_le();  // fc hash

        n.fc0_bias[st].resize(SL2);
        for (int i = 0; i < SL2; ++i) n.fc0_bias[st][i] = int32_t(r.read_u32_le());
        n.fc0_w[st].resize(size_t(SL2) * g_sL1);
        if (!r.read_raw(n.fc0_w[st].size(), n.fc0_w[st].data())) return false;

        n.fc1_bias[st].resize(SL3);
        for (int i = 0; i < SL3; ++i) n.fc1_bias[st][i] = int32_t(r.read_u32_le());
        n.fc1_w[st].resize(size_t(SL3) * SL2_PAD);
        if (!r.read_raw(n.fc1_w[st].size(), n.fc1_w[st].data())) return false;

        n.fc2_bias[st].resize(1);
        n.fc2_bias[st][0] = int32_t(r.read_u32_le());
        n.fc2_w[st].resize(size_t(SL3));
        if (!r.read_raw(n.fc2_w[st].size(), n.fc2_w[st].data())) return false;
    }

    if (!r.ok || r.pos != r.size) return false;

    g_snet    = std::move(n);
    g_sloaded = true;
    g_spath   = path;
    return true;
}

bool small_loaded() { return g_sloaded; }
const std::string& small_path() { return g_spath; }
bool big_net_loaded() { return g_loaded; }
int evaluate_small(const Board& b) { return g_sloaded ? full_small(b) : 0; }
void set_big_net_ply(int v) { g_bignet_ply = std::clamp(v, 0, 64); }
int  big_net_ply() { return g_bignet_ply; }
void set_big_net_pv(int v) { g_bignet_pv = v ? 1 : 0; }
int  big_net_pv() { return g_bignet_pv; }

bool is_loaded() { return g_loaded; }
const std::string& loaded_path() { return g_path; }

namespace {

// ---- Disabled experiment: lazy (on-demand) weight-row application ----
// Measured slower than the eager path: materialize_top() re-copies the
// accumulator and re-applies deltas along the path, while the real bottleneck
// turned out to be the feature enumeration in acc_push, not the row applies.
// Kept here (commented out) for reference.
#if 0
// Apply one node's pending feature deltas to its (already copied) accumulator.
void apply_pending(AccEntry& e) {
    const int16_t* psqW    = g_net.weights.data();
    const int32_t* psqtW   = g_net.psqtWeights.data();
    const int8_t*  thrW    = g_net.threatWeights.data();
    const int32_t* thrPsqtW = g_net.threatPsqt.data();

    for (int p = 0; p < 2; ++p) {
        AccT*    acc = e.acc[p];
        int32_t* pst = e.psqt[p];

        const IndexList& pr = e.psqRem[p];
        const IndexList& pa = e.psqAdd[p];
        for (int i = 0; i < pr.n; ++i) apply_feature(psqW, psqtW, pr.v[i], -1, acc, pst);
        for (int i = 0; i < pa.n; ++i) apply_feature(psqW, psqtW, pa.v[i], +1, acc, pst);

        const IndexList& tr = e.thrRem[p];  // sorted
        const IndexList& ta = e.thrAdd[p];  // sorted
        int i = 0, j = 0;
        while (i < tr.n && j < ta.n) {
            if (tr.v[i] == ta.v[j]) { ++i; ++j; }
            else if (tr.v[i] < ta.v[j]) apply_feature(thrW, thrPsqtW, tr.v[i++], -1, acc, pst);
            else                        apply_feature(thrW, thrPsqtW, ta.v[j++], +1, acc, pst);
        }
        for (; i < tr.n; ++i) apply_feature(thrW, thrPsqtW, tr.v[i], -1, acc, pst);
        for (; j < ta.n; ++j) apply_feature(thrW, thrPsqtW, ta.v[j], +1, acc, pst);
    }
}

// Bring the top entry up to date by applying only the pending deltas of the
// path back to the nearest already-materialized ancestor.  Subtrees that are
// never evaluated therefore never touch the huge weight rows.
void materialize_top() {
    const int t = g_acc.size - 1;
    if (t < 0 || g_acc.entries[t].materialized) return;
    int k = t - 1;
    while (k > 0 && !g_acc.entries[k].materialized) --k;
    const uint64_t t0 = rdtsc_now();
    for (int i = k + 1; i <= t; ++i) {
        AccEntry&       e   = g_acc.entries[i];
        const AccEntry& par = g_acc.entries[i - 1];
        std::memcpy(e.acc, par.acc, sizeof(e.acc));
        std::memcpy(e.psqt, par.psqt, sizeof(e.psqt));
        apply_pending(e);
        e.materialized = true;
    }
    g_cyc_apply += rdtsc_now() - t0;
}
#endif

}  // namespace

// ---- Incremental accumulator API (driven by the searcher) ----

namespace {

// Full PSQ build of the small net's accumulator (g_sL1 wide).
void init_small_entry(const Board& b, AccEntry& e) {
    for (int persp = 0; persp < 2; ++persp) {
        const int ksq = int(b.king_square(Color(persp)));
        for (int i = 0; i < g_sL1; ++i) e.accS[persp][i] = g_snet.biases[i];
        for (int k = 0; k < PSQTBuckets; ++k) e.psqtS[persp][k] = 0;
        uint64_t occ = b.occupancy();
        while (occ) {
            const int s = lsb_sq(occ);
            occ &= occ - 1;
            const int sfp = sf_piece(b.piece_on(Square(s)));
            if (!sfp) continue;
            apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                              halfka_make_index(persp, s, sfp, ksq), +1, e.accS[persp],
                              e.psqtS[persp], g_sL1);
        }
    }
}

// Full build of the big net's accumulator (PSQ + threats/pairs) and of the
// active threat/pair feature lists that later pushes diff against.
void init_big_full(const Board& b, AccEntry& e) {
    IndexList psq, empty;
    for (int persp = 0; persp < 2; ++persp) {
        const int ksq = int(b.king_square(Color(persp)));
        for (int i = 0; i < L1; ++i) e.acc[persp][i] = g_net.biases[i];
        for (int k = 0; k < PSQTBuckets; ++k) e.psqt[persp][k] = 0;

        psq.clear();
        uint64_t occ = b.occupancy();
        while (occ) {
            const int s = lsb_sq(occ);
            occ &= occ - 1;
            const int sfp = sf_piece(b.piece_on(Square(s)));
            if (!sfp) continue;
            psq.push(halfka_make_index(persp, s, sfp, ksq));
        }
        psq.sort();
        apply_diff(g_net.weights.data(), g_net.psqtWeights.data(), empty, psq,
                   e.acc[persp], e.psqt[persp]);

        if (!g_threat_inc) continue;  // legacy: accumulator is PSQ-only

        e.thr[persp].clear();
        append_active_all(b, persp, e.thr[persp]);
        e.thr[persp].sort();
        apply_diff(g_net.threatWeights.data(), g_net.threatPsqt.data(), empty,
                   e.thr[persp], e.acc[persp], e.psqt[persp]);
    }
}

}  // namespace

void acc_set_root(const Board& b) {
    if (!g_loaded || !g_sloaded) return;
    static const bool once = []() {
        g_threat_inc = (std::getenv("NNUE_LEGACY_BIGNET") == nullptr);
        return true;
    }();
    (void)once;
    acc_reset_stats();
    if (g_acc.entries.empty()) g_acc.entries.resize(MaxAccPly);
    g_acc.size  = 1;
    AccEntry& e = g_acc.entries[0];
    e.key      = b.key();
    e.valid    = true;
    e.big      = true;
    e.bigValid = true;
    init_small_entry(b, e);
    init_big_full(b, e);
}

void acc_push(const Board& b, bool big) {
    if (!g_loaded || !g_sloaded || g_acc.size <= 0 || g_acc.size >= int(g_acc.entries.size()))
        return;
    AccEntry& parent = g_acc.entries[g_acc.size - 1];
    AccEntry& child  = g_acc.entries[g_acc.size];

    // Small net: always incremental (cheap HalfKAv2_hm PSQ rows). Only the
    // first g_sL1 lanes of each perspective are live, so copy just those
    // instead of the full 1024-wide storage.
    {
        const size_t bytes = size_t(g_sL1) * sizeof(AccT);
        std::memcpy(child.accS[0], parent.accS[0], bytes);
        std::memcpy(child.accS[1], parent.accS[1], bytes);
    }
    std::memcpy(child.psqtS, parent.psqtS, sizeof(child.psqtS));
    child.key      = b.key();
    child.valid    = true;
    child.big      = big;
    child.bigValid = false;

    const uint64_t tp0 = rdtsc_now();

    const HistoryEntry& he       = b.history[b.history_ply - 1];
    const Move          m        = he.move;
    const Piece         captured = Piece(he.captured);

    if (m == Move::none()) {  // null move: feature set unchanged
        if (big && parent.bigValid) {
            std::memcpy(child.acc, parent.acc, sizeof(child.acc));
            std::memcpy(child.psqt, parent.psqt, sizeof(child.psqt));
            child.thr[0]   = parent.thr[0];
            child.thr[1]   = parent.thr[1];
            child.bigValid = true;
        }
        ++g_acc.size;
        ++g_push_count;
        return;
    }

    const int    flags     = m.flags();
    const Square from      = m.from();
    const Square to        = m.to();
    const Color  mover     = ~b.stm;
    const Piece  childAtTo = b.board[to];

    if (type_of(childAtTo) == KING) {  // king/castle: rekey the mover's perspective
        const size_t sbytes = size_t(g_sL1) * sizeof(AccT);
        std::memcpy(child.accS[0], parent.accS[0], sbytes);
        std::memcpy(child.accS[1], parent.accS[1], sbytes);
        std::memcpy(child.psqtS, parent.psqtS, sizeof(child.psqtS));

        if (m.is_castle()) {
            init_small_entry(b, child);  // king and rook both moved: full rebuild
        } else {
            // The opponent's view is keyed to the opponent king, which did not
            // move, so only the moved king's own feature changes there.
            const int otherP   = int(mover) ^ 1;
            const int otherKsq = int(b.king_square(Color(otherP)));
            const int ksfp     = sf_piece(make_piece(mover, KING));
            apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                              halfka_make_index(otherP, from, ksfp, otherKsq), -1,
                              child.accS[otherP], child.psqtS[otherP], g_sL1);
            apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                              halfka_make_index(otherP, to, ksfp, otherKsq), +1,
                              child.accS[otherP], child.psqtS[otherP], g_sL1);
            if (flags & CAPTURE) {  // king capture: drop the captured piece too
                apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                                  halfka_make_index(otherP, to, sf_piece(captured), otherKsq), -1,
                                  child.accS[otherP], child.psqtS[otherP], g_sL1);
            }

            // The mover's view is fully rekeyed by its king square.
            const int moverP   = int(mover);
            const int moverKsq = int(b.king_square(Color(moverP)));
            for (int i = 0; i < g_sL1; ++i) child.accS[moverP][i] = g_snet.biases[i];
            for (int k = 0; k < PSQTBuckets; ++k) child.psqtS[moverP][k] = 0;
            uint64_t occ = b.occupancy();
            while (occ) {
                const int s = lsb_sq(occ);
                occ &= occ - 1;
                const int sfp = sf_piece(b.piece_on(Square(s)));
                if (!sfp) continue;
                apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                                  halfka_make_index(moverP, s, sfp, moverKsq), +1,
                                  child.accS[moverP], child.psqtS[moverP], g_sL1);
            }
        }
        if (big && parent.bigValid) {
            init_big_full(b, child);
            child.bigValid = true;
        }
        ++g_acc.size;
        ++g_push_count;
        return;
    }

    const Piece  parentMoved = (flags & PROMOTION) ? make_piece(mover, PAWN) : childAtTo;
    const Square capSq       = (flags == EP_CAPTURE) ? Square(to ^ 8) : to;
    const bool   isCastle    = (flags == KING_CASTLE || flags == QUEEN_CASTLE);

    const View cv = make_view(b);
    View       pv = cv;
    pv.pcs[to]   = (flags == EP_CAPTURE) ? NO_PIECE : captured;
    pv.pcs[from] = parentMoved;
    if (flags == EP_CAPTURE) pv.pcs[capSq] = captured;

    Square rfrom = SQ_NONE, rto = SQ_NONE;
    if (isCastle) {
        const Piece rook = make_piece(mover, ROOK);
        if (flags == KING_CASTLE) { rfrom = Square(7 + mover * 56); rto = Square(5 + mover * 56); }
        else                      { rfrom = Square(0 + mover * 56); rto = Square(3 + mover * 56); }
        pv.pcs[rto]   = NO_PIECE;
        pv.pcs[rfrom] = rook;
    }

    Bitboard changed = (Bitboard(1) << from) | (Bitboard(1) << to);
    if (flags == EP_CAPTURE) changed |= Bitboard(1) << capSq;
    if (isCastle) changed |= (Bitboard(1) << rfrom) | (Bitboard(1) << rto);

    // Small net PSQ delta.
    for (int persp = 0; persp < 2; ++persp) {
        const int ksq = int(b.king_square(Color(persp)));
        for (Bitboard x = changed; x;) {
            const int s = lsb_sq(x);
            x &= x - 1;
            const Piece pp = pv.pcs[s];
            const Piece cp = cv.pcs[s];
            if (pp == cp) continue;
            if (pp != NO_PIECE)
                apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                                  halfka_make_index(persp, s, sf_piece(pp), ksq), -1,
                                  child.accS[persp], child.psqtS[persp], g_sL1);
            if (cp != NO_PIECE)
                apply_feature_len(g_snet.weights.data(), g_snet.psqtWeights.data(),
                                  halfka_make_index(persp, s, sf_piece(cp), ksq), +1,
                                  child.accS[persp], child.psqtS[persp], g_sL1);
        }
    }

    // Big net: only maintained along requested big paths. Boundary big nodes
    // (a big node whose parent was not big) fall back to a full refresh in
    // evaluate(), so non-big subtrees never pay for the 1024-wide accumulator.
    if (big && parent.bigValid) {
        std::memcpy(child.acc, parent.acc, sizeof(child.acc));
        std::memcpy(child.psqt, parent.psqt, sizeof(child.psqt));

        for (int persp = 0; persp < 2; ++persp) {
            const int ksq = int(b.king_square(Color(persp)));
            for (Bitboard x = changed; x;) {
                const int s = lsb_sq(x);
                x &= x - 1;
                const Piece pp = pv.pcs[s];
                const Piece cp = cv.pcs[s];
                if (pp == cp) continue;
                if (pp != NO_PIECE)
                    apply_feature(g_net.weights.data(), g_net.psqtWeights.data(),
                                  halfka_make_index(persp, s, sf_piece(pp), ksq), -1,
                                  child.acc[persp], child.psqt[persp]);
                if (cp != NO_PIECE)
                    apply_feature(g_net.weights.data(), g_net.psqtWeights.data(),
                                  halfka_make_index(persp, s, sf_piece(cp), ksq), +1,
                                  child.acc[persp], child.psqt[persp]);
            }
        }

        // Threat/pair delta: enumerate the child's active features and apply
        // only the symmetric difference against the parent's stored list.
        if (g_threat_inc) {
            IndexList thr;
            for (int persp = 0; persp < 2; ++persp) {
                thr.clear();
                append_active_all(b, persp, thr);
                thr.sort();
                apply_diff(g_net.threatWeights.data(), g_net.threatPsqt.data(),
                           parent.thr[persp], thr, child.acc[persp], child.psqt[persp]);
                child.thr[persp] = thr;
            }
        }

        child.bigValid = true;
    }

    g_cyc_apply += rdtsc_now() - tp0;
    ++g_acc.size;
    ++g_push_count;
}

void acc_pop() {
    if (g_acc.size > 1) --g_acc.size;
}

int evaluate(const Board& b) {
    if (!g_loaded && !g_sloaded) return 0;

    static const bool verify = (std::getenv("NNUE_VERIFY_INC") != nullptr);

    if (g_loaded && g_sloaded && g_acc.size > 0) {
        const AccEntry& e = g_acc.entries[g_acc.size - 1];
        if (e.valid && e.key == b.key()) {
            const int stm = int(b.side_to_move());
            const int cnt = constexpr_popcount(b.occupancy());
            ++g_inc_evals;
            if (e.big) ++g_big_evals;
            const uint64_t tf0 = rdtsc_now();
            int inc;
            if (e.big && !e.bigValid) {
                // Boundary big node: no incremental accumulator available.
                inc = evaluate_full(b);
            } else if (e.big) {
                if (g_threat_inc) {
                    // Big net: full accumulator kept incrementally along big paths.
                    inc = finish(g_net, stm, cnt, e.acc, e.psqt);
                } else {
                    // Legacy: accumulator is PSQ-only; add threats/pairs here.
                    static thread_local AccT    accB[2][L1];
                    static thread_local int32_t psqtB[2][PSQTBuckets];
                    std::memcpy(accB, e.acc, sizeof(accB));
                    std::memcpy(psqtB, e.psqt, sizeof(psqtB));
                    for (int persp = 0; persp < 2; ++persp) {
                        IndexList thr;
                        threat_append_active(b, persp, thr);
                        pair_append_active(b, persp, thr);
                        for (int i = 0; i < thr.n; ++i)
                            apply_feature(g_net.threatWeights.data(), g_net.threatPsqt.data(),
                                          thr.v[i], +1, accB[persp], psqtB[persp]);
                    }
                    inc = finish(g_net, stm, cnt, accB, psqtB);
                }
            } else {
                inc = finish_small(g_snet, stm, cnt, e.accS, e.psqtS);
            }
            g_cyc_finish += rdtsc_now() - tf0;
            if (!verify) return inc;
            const int full = e.big ? evaluate_full(b) : full_small(b);
            if (inc != full) {
                char dbgfen[128] = {0};
                b.to_fen(dbgfen);
                std::fprintf(stderr, "NNUE incremental mismatch: inc=%d full=%d fen=%s\n", inc,
                             full, dbgfen);
            }
            return full;
        }
    }
    ++g_full_evals;
    if (g_loaded) return evaluate_full(b);
    return full_small(b);
}

AccStats acc_stats() {
    return AccStats{g_inc_evals.load(std::memory_order_relaxed),
                    g_full_evals.load(std::memory_order_relaxed),
                    g_push_count.load(std::memory_order_relaxed),
                    g_diff_applies.load(std::memory_order_relaxed),
                    g_big_evals.load(std::memory_order_relaxed)};
}

void acc_reset_stats() {
    g_inc_evals.store(0, std::memory_order_relaxed);
    g_full_evals.store(0, std::memory_order_relaxed);
    g_push_count.store(0, std::memory_order_relaxed);
    g_diff_applies.store(0, std::memory_order_relaxed);
    g_big_evals.store(0, std::memory_order_relaxed);
    g_cyc_build = g_cyc_apply = g_cyc_finish = 0;
    g_cyc_build_append = g_cyc_build_sort = 0;
}

void acc_cycles(uint64_t& build, uint64_t& apply, uint64_t& finish) {
    build  = g_cyc_build;
    apply  = g_cyc_apply;
    finish = g_cyc_finish;
}

void acc_build_cycles(uint64_t& append, uint64_t& sort) {
    append = g_cyc_build_append;
    sort   = g_cyc_build_sort;
}

uint64_t tsc_now() { return rdtsc_now(); }

}  // namespace nnue_full
