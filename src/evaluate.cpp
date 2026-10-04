#include "evaluate.h"

#include "bitboard.h"
#include "nnue.h"

#include <cstdio>

namespace eval {

namespace {

constexpr int P = 100, N = 320, B = 330, R = 500, Q = 900, K = 0;

constexpr int PIECE_VALUE_LOCAL[PIECE_TYPE_NB] = { P, N, B, R, Q, K };

constexpr int PST_PAWN[SQ_NB] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
};

constexpr int PST_KNIGHT[SQ_NB] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
};

constexpr int PST_BISHOP[SQ_NB] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
};

constexpr int PST_ROOK[SQ_NB] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
};

constexpr int PST_QUEEN[SQ_NB] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
};

constexpr int PST_KING[SQ_NB] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
};

constexpr int PST_KING_END[SQ_NB] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
};

int material(const Board& b) {
    int score = 0;
    for (int c = 0; c < COLOR_NB; ++c) {
        int sign = c == WHITE ? 1 : -1;
        for (int pt = PAWN; pt <= QUEEN; ++pt)
            score += sign * PIECE_VALUE_LOCAL[pt] * __builtin_popcountll(b.pieces(Color(c), PieceType(pt)));
    }
    return score;
}

}  // namespace

void init() { nnue::init(); }

bool try_load_nnue(const char* path) { return nnue::load(path); }
bool try_load_nnue_small(const char* path) { return nnue::load_small(path); }
bool using_nnue() { return nnue::is_loaded(); }
bool using_nnue_small() { return nnue::small_loaded(); }
void nnue_set_big_net_ply(int v) { nnue::set_big_net_ply(v); }
int  nnue_big_net_ply() { return nnue::big_net_ply(); }
void nnue_set_big_net_pv(int v) { nnue::set_big_net_pv(v); }
int  nnue_big_net_pv() { return nnue::big_net_pv(); }

void nnue_set_root(const Board& b) {
    if (nnue::is_loaded() && nnue::small_loaded()) nnue::acc_set_root(b);
}
void nnue_on_make(const Board& b, bool big) {
    if (nnue::is_loaded() && nnue::small_loaded()) nnue::acc_push(b, big);
}
void nnue_on_unmake() {
    if (nnue::is_loaded() && nnue::small_loaded()) nnue::acc_pop();
}

int evaluate(const Board& b) {
    if (nnue::is_loaded())
        return nnue::evaluate(b);

    int score = material(b);

    for (int c = 0; c < COLOR_NB; ++c) {
        int sign = c == WHITE ? 1 : -1;
        for (int pt = PAWN; pt <= QUEEN; ++pt) {
            Bitboard bb = b.pieces(Color(c), PieceType(pt));
            while (bb) {
                Square s = Square(__builtin_ctzll(bb));
                bb &= bb - 1;
                int idx = c == WHITE ? s : (s ^ 56);
                switch (pt) {
                    case PAWN:   score += sign * PST_PAWN[idx]; break;
                    case KNIGHT: score += sign * PST_KNIGHT[idx]; break;
                    case BISHOP: score += sign * PST_BISHOP[idx]; break;
                    case ROOK:   score += sign * PST_ROOK[idx]; break;
                    case QUEEN:  score += sign * PST_QUEEN[idx]; break;
                    default: break;
                }
            }
        }
    }

    bool endgame = __builtin_popcountll(b.pieces(QUEEN)) +
                   __builtin_popcountll(b.pieces(ROOK)) <= 2;
    for (int c = 0; c < COLOR_NB; ++c) {
        int sign = c == WHITE ? 1 : -1;
        Square ks = b.king_square(Color(c));
        int idx = c == WHITE ? ks : (ks ^ 56);
        score += sign * (endgame ? PST_KING_END[idx] : PST_KING[idx]);
    }

    return b.side_to_move() == WHITE ? score : -score;
}

}  // namespace eval
