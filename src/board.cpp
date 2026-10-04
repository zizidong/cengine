#include "board.h"

#include "attack.h"
#include "bitboard.h"
#include "zobrist.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdio>

Board::Board() { reset(); }

void Board::reset() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::to_fen(char* out) const {
    int o = 0;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[r * 8 + f];
            if (p == NO_PIECE) {
                ++empty;
            } else {
                if (empty) { out[o++] = '0' + empty; empty = 0; }
                static const char chars[] = "PNBRQKpnbrqk";
                out[o++] = chars[p];
            }
        }
        if (empty) out[o++] = '0' + empty;
        if (r) out[o++] = '/';
    }
    out[o++] = ' ';
    out[o++] = (stm == WHITE) ? 'w' : 'b';
    out[o++] = ' ';
    int cr = castling_rights;
    if (!cr) {
        out[o++] = '-';
    } else {
        if (cr & WHITE_KING_SIDE)  out[o++] = 'K';
        if (cr & WHITE_QUEEN_SIDE) out[o++] = 'Q';
        if (cr & BLACK_KING_SIDE)  out[o++] = 'k';
        if (cr & BLACK_QUEEN_SIDE) out[o++] = 'q';
    }
    out[o++] = ' ';
    if (ep == SQ_NONE) {
        out[o++] = '-';
    } else {
        out[o++] = 'a' + FILE_OF(ep);
        out[o++] = '1' + RANK_OF(ep);
    }
    out[o++] = ' ';
    out[o++] = '0';
    out[o++] = ' ';
    out[o++] = '1';
    out[o] = '\0';
}

namespace {

}  // namespace

void Board::set_fen(const char* fen) {
    std::fill(board, board + SQ_NB, NO_PIECE);
    for (int i = 0; i < PIECE_TYPE_NB; ++i) by_type[i] = 0;
    by_color[WHITE] = by_color[BLACK] = 0;
    occ = 0;
    castling_rights = 0;
    ep = SQ_NONE;
    fifty = 0;
    ply = 0;
    state_key = 0;
    history_ply = 0;

    const char* p = fen;
    int sq = SQ_A8;
    while (*p && *p != ' ') {
        char c = *p++;
        if (c == '/') {
            sq -= 16;
        } else if (c >= '1' && c <= '8') {
            sq += c - '0';
        } else {
            Piece piece = NO_PIECE;
            switch (c) {
                case 'P': piece = W_PAWN; break;
                case 'N': piece = W_KNIGHT; break;
                case 'B': piece = W_BISHOP; break;
                case 'R': piece = W_ROOK; break;
                case 'Q': piece = W_QUEEN; break;
                case 'K': piece = W_KING; break;
                case 'p': piece = B_PAWN; break;
                case 'n': piece = B_KNIGHT; break;
                case 'b': piece = B_BISHOP; break;
                case 'r': piece = B_ROOK; break;
                case 'q': piece = B_QUEEN; break;
                case 'k': piece = B_KING; break;
            }
            board[sq] = piece;
            by_type[type_of(piece)] |= 1ULL << sq;
            by_color[color_of(piece)] |= 1ULL << sq;
            occ |= 1ULL << sq;
            state_key ^= zobrist::psq[piece][sq];
            if (type_of(piece) == KING) king_sq[color_of(piece)] = Square(sq);
            ++sq;
        }
    }

    while (*p == ' ') ++p;
    stm = (*p == 'w') ? WHITE : BLACK;
    ++p;
    while (*p == ' ') ++p;

    while (*p && *p != ' ') {
        switch (*p++) {
            case 'K': castling_rights |= WHITE_KING_SIDE; break;
            case 'Q': castling_rights |= WHITE_QUEEN_SIDE; break;
            case 'k': castling_rights |= BLACK_KING_SIDE; break;
            case 'q': castling_rights |= BLACK_QUEEN_SIDE; break;
            case '-': break;
        }
    }
    state_key ^= zobrist::castling[castling_rights];
    ++p;

    if (*p && *p != ' ' && *p != '-') {
        int file = (*p++) - 'a';
        int rank = (*p++) - '1';
        ep = Square(rank * 8 + file);
        state_key ^= zobrist::ep_file[FILE_OF(ep)];
    }
    while (*p && *p != ' ') ++p;

    fifty = atoi(p);
    if (stm == BLACK) state_key ^= zobrist::side;
}

Bitboard Board::attackers_to(Square s, Bitboard occ) const {
    return (by_type[PAWN] & by_color[WHITE] & bb::pawn_attacks[BLACK][s])
         | (by_type[PAWN] & by_color[BLACK] & bb::pawn_attacks[WHITE][s])
         | (by_type[KNIGHT] & bb::knight_attacks[s])
         | (by_type[BISHOP] & bb::pseudo_attacks(BISHOP, s, occ))
         | (by_type[ROOK]   & bb::pseudo_attacks(ROOK, s, occ))
         | (by_type[QUEEN]  & bb::pseudo_attacks(QUEEN, s, occ))
         | (by_type[KING]   & bb::king_attacks[s]);
}

Bitboard Board::checkers() const {
    return attackers_to(king_sq[stm], occupancy()) & by_color[non_us()];
}

Bitboard Board::checkers(Color c) const {
    return attackers_to(king_sq[c], occupancy()) & by_color[~c];
}

// Direction step from a to b, or 0 if not aligned.
static int step_between(Square a, Square b) {
    const int df = int(FILE_OF(b)) - int(FILE_OF(a));
    const int dr = int(RANK_OF(b)) - int(RANK_OF(a));
    if (df == 0 && dr != 0) return dr > 0 ? 8 : -8;
    if (dr == 0 && df != 0) return df > 0 ? 1 : -1;
    if (df == dr && df != 0) return df > 0 ? 9 : -9;    // up-right / down-left
    if (df == -dr && df != 0) return df > 0 ? -7 : 7;   // down-right / up-left
    return 0;
}

// Squares strictly between a and b (0 if not aligned).
static Bitboard ray_between(Square a, Square b) {
    const int st = step_between(a, b);
    if (!st) return 0;
    Bitboard m = 0;
    for (int s = int(a) + st; s != int(b); s += st) {
        if (s < 0 || s >= SQ_NB) return 0;
        m |= 1ULL << s;
    }
    return m;
}

// For a piece pinned to `ksq`: the squares it may move to while still blocking
// (from itself, the squares beyond it away from the king, and the squares back
// toward the king).
static Bitboard pinned_allowed(Square ksq, Square from) {
    const int st = step_between(ksq, from);
    if (!st) return 0;
    Bitboard m = 1ULL << int(from);
    for (int s = int(from) + st; s >= 0 && s < SQ_NB; s += st) {
        if (step_between(from, Square(s)) != st) break;  // file wrap guard
        m |= 1ULL << s;
    }
    return m | ray_between(ksq, from);
}

Bitboard Board::pinned_pieces(Color c) const {
    const Square   ksq    = king_sq[c];
    const Bitboard theirR = (by_type[ROOK] | by_type[QUEEN]) & by_color[~c];
    const Bitboard theirB = (by_type[BISHOP] | by_type[QUEEN]) & by_color[~c];
    Bitboard       cand   = (theirR & bb::pseudo_attacks(ROOK, ksq, 0))
                        | (theirB & bb::pseudo_attacks(BISHOP, ksq, 0));
    Bitboard pinned = 0;
    while (cand) {
        const Square s = Square(__builtin_ctzll(cand));
        cand &= cand - 1;
        const Bitboard between = ray_between(ksq, s) & occ;
        if (between && !(between & (between - 1)) && (between & by_color[c])) pinned |= between;
    }
    return pinned;
}

bool Board::legal_precheck(Move m, Bitboard pinned) const {
    const Square from = m.from();
    const Square to   = m.to();
    const Piece  p    = board[from];

    if (type_of(p) == KING) {
        if (m.is_castle()) return true;  // movegen already verified the path
        const Bitboard occAfter = (occ ^ (1ULL << from)) | (1ULL << to);
        return !(attackers_to(to, occAfter) & by_color[~stm]);
    }
    if (m.is_ep()) return true;  // rare; verified after the move
    if (pinned & (1ULL << from))
        return (pinned_allowed(king_sq[stm], from) & (1ULL << to)) != 0;
    return true;
}

bool Board::gives_check(Move, Bitboard) const {
    return false;
}

bool Board::is_repetition() const {
    int end = std::min(fifty, history_ply);
    for (int i = 4; i <= end; i += 2)
        if (history[history_ply - i].key == state_key) return true;
    return false;
}

// Dead-draw material: K vs K, K+minor vs K, and KB vs KB on one colour.
bool Board::is_insufficient_material() const {
    if (by_type[PAWN] | by_type[ROOK] | by_type[QUEEN]) return false;
    const int wb = __builtin_popcountll(by_type[BISHOP] & by_color[WHITE]);
    const int wn = __builtin_popcountll(by_type[KNIGHT] & by_color[WHITE]);
    const int bb = __builtin_popcountll(by_type[BISHOP] & by_color[BLACK]);
    const int bn = __builtin_popcountll(by_type[KNIGHT] & by_color[BLACK]);
    const int wm = wb + wn, bm = bb + bn;
    if (wm == 0 && bm == 0) return true;
    if (wm + bm == 1) return true;
    if (wb == 1 && bb == 1 && wm == 1 && bm == 1) {
        const Square ws = Square(__builtin_ctzll(by_type[BISHOP] & by_color[WHITE]));
        const Square bs = Square(__builtin_ctzll(by_type[BISHOP] & by_color[BLACK]));
        return ((FILE_OF(ws) + RANK_OF(ws)) & 1) == ((FILE_OF(bs) + RANK_OF(bs)) & 1);
    }
    return false;
}

bool Board::see_ge(Move m, int threshold) const {
    static constexpr int SEE_VALUE[PIECE_TYPE_NB] = { 100, 300, 300, 500, 900, 20000 };

    Square from = m.from();
    Square to   = m.to();

    if (m.is_castle()) return true;
    if (m.is_ep()) {
        // en passant: winning a pawn for a pawn is at least a pawn in the common case
        return threshold <= 0;
    }

    int swap = SEE_VALUE[type_on(to)] - threshold;
    if (swap < 0) return false;
    swap = SEE_VALUE[type_on(from)] - swap;
    if (swap <= 0) return true;

    Bitboard occ = occupancy() ^ (1ULL << from) ^ (1ULL << to);
    Color mover = stm;          // side that played m
    Color side  = ~mover;       // side to recapture
    Bitboard attackers = attackers_to(to, occ) & occ;

    int res = 1;

    while (true) {
        attackers &= occ;
        Bitboard stmAttackers = attackers & by_color[side];
        if (!stmAttackers) break;

        // smallest attacking piece
        PieceType pt = PAWN;
        Bitboard bb = 0;
        for (PieceType cand = PAWN; cand <= KING; cand = PieceType(cand + 1)) {
            bb = stmAttackers & by_type[cand];
            if (bb) { pt = cand; break; }
        }
        if (!bb) break;

        if ((swap = SEE_VALUE[pt] - swap) < res) break;

        occ ^= bb & (~bb + 1);  // remove the least significant attacker
        if (pt == PAWN || pt == BISHOP || pt == QUEEN)
            attackers |= bb::pseudo_attacks(BISHOP, to, occ) & (by_type[BISHOP] | by_type[QUEEN]);
        if (pt == ROOK || pt == QUEEN)
            attackers |= bb::pseudo_attacks(ROOK, to, occ) & (by_type[ROOK] | by_type[QUEEN]);

        side = ~side;
        res = -res;
    }
    return res > 0;
}

void Board::make_move(Move m) {
    HistoryEntry& he = history[history_ply++];
    he.move = m;
    he.captured = NO_PIECE;
    he.castling = uint8_t(castling_rights);
    he.ep = ep;
    he.fifty = uint8_t(fifty);
    he.key = state_key;

    Square from = m.from();
    Square to = m.to();
    Piece piece = board[from];
    int flags = m.flags();

    state_key ^= zobrist::castling[castling_rights];
    state_key ^= zobrist::ep_file[FILE_OF(ep)];

    ep = SQ_NONE;

    if (flags & CAPTURE) {
        Square cap_sq = (flags == EP_CAPTURE) ? Square(to ^ 8) : to;
        Piece cap = board[cap_sq];
        he.captured = cap;
        state_key ^= zobrist::psq[cap][cap_sq];
        by_type[type_of(cap)] ^= 1ULL << cap_sq;
        by_color[color_of(cap)] ^= 1ULL << cap_sq;
        occ ^= 1ULL << cap_sq;
        board[cap_sq] = NO_PIECE;
    }

    by_type[type_of(piece)] ^= (1ULL << from) | (1ULL << to);
    by_color[color_of(piece)] ^= (1ULL << from) | (1ULL << to);
    occ ^= (1ULL << from) | (1ULL << to);
    state_key ^= zobrist::psq[piece][from] ^ zobrist::psq[piece][to];
    board[from] = NO_PIECE;
    board[to] = piece;

    PieceType pt = type_of(piece);

    if (pt == PAWN) {
        if (flags == DOUBLE_PAWN) {
            ep = Square((from + to) / 2);
            state_key ^= zobrist::ep_file[FILE_OF(ep)];
        } else if (flags & PROMOTION) {
            Piece promoted = make_piece(stm, m.promo_type());
            by_type[PAWN] ^= 1ULL << to;
            by_type[m.promo_type()] ^= 1ULL << to;
            state_key ^= zobrist::psq[piece][to] ^ zobrist::psq[promoted][to];
            board[to] = promoted;
        }
    } else if (pt == KING) {
        king_sq[stm] = to;
        if (flags == KING_CASTLE) {
            Square rfrom = Square(7 + (stm * 56));
            Square rto = Square(5 + (stm * 56));
            Piece rook = board[rfrom];
            by_type[ROOK] ^= (1ULL << rfrom) | (1ULL << rto);
            by_color[color_of(rook)] ^= (1ULL << rfrom) | (1ULL << rto);
            occ ^= (1ULL << rfrom) | (1ULL << rto);
            state_key ^= zobrist::psq[rook][rfrom] ^ zobrist::psq[rook][rto];
            board[rfrom] = NO_PIECE;
            board[rto] = rook;
        } else if (flags == QUEEN_CASTLE) {
            Square rfrom = Square(0 + (stm * 56));
            Square rto = Square(3 + (stm * 56));
            Piece rook = board[rfrom];
            by_type[ROOK] ^= (1ULL << rfrom) | (1ULL << rto);
            by_color[color_of(rook)] ^= (1ULL << rfrom) | (1ULL << rto);
            occ ^= (1ULL << rfrom) | (1ULL << rto);
            state_key ^= zobrist::psq[rook][rfrom] ^ zobrist::psq[rook][rto];
            board[rfrom] = NO_PIECE;
            board[rto] = rook;
        }
    }

    int cr = castling_rights;
    if (pt == KING)
        cr &= ~(stm == WHITE ? (WHITE_KING_SIDE | WHITE_QUEEN_SIDE) : (BLACK_KING_SIDE | BLACK_QUEEN_SIDE));
    if (pt == ROOK) {
        if (from == SQ_A1) cr &= ~WHITE_QUEEN_SIDE;
        if (from == SQ_H1) cr &= ~WHITE_KING_SIDE;
        if (from == SQ_A8) cr &= ~BLACK_QUEEN_SIDE;
        if (from == SQ_H8) cr &= ~BLACK_KING_SIDE;
    }
    if (flags & CAPTURE) {
        if (to == SQ_A1) cr &= ~WHITE_QUEEN_SIDE;
        if (to == SQ_H1) cr &= ~WHITE_KING_SIDE;
        if (to == SQ_A8) cr &= ~BLACK_QUEEN_SIDE;
        if (to == SQ_H8) cr &= ~BLACK_KING_SIDE;
    }
    castling_rights = cr;
    state_key ^= zobrist::castling[castling_rights];

    fifty = (pt == PAWN || (flags & CAPTURE)) ? 0 : fifty + 1;
    ++ply;
    stm = ~stm;
    state_key ^= zobrist::side;
}

void Board::unmake_move() {
    HistoryEntry& he = history[--history_ply];
    stm = ~stm;
    --ply;

    Square from = he.move.from();
    Square to = he.move.to();
    int flags = he.move.flags();
    Piece piece = board[to];
    PieceType pt = flags & PROMOTION ? PAWN : type_of(piece);

    state_key = he.key;

    if (flags & PROMOTION) {
        PieceType promo = type_of(board[to]);
        by_type[promo] ^= 1ULL << to;
        by_type[PAWN] |= 1ULL << to;
    }
    piece = make_piece(stm, pt);

    by_type[pt] ^= (1ULL << from) | (1ULL << to);
    by_color[color_of(piece)] ^= (1ULL << from) | (1ULL << to);
    occ ^= (1ULL << from) | (1ULL << to);
    board[from] = piece;
    board[to] = NO_PIECE;

    if (pt == KING) {
        king_sq[stm] = from;
        if (flags == KING_CASTLE) {
            Square rfrom = Square(5 + (stm * 56));
            Square rto = Square(7 + (stm * 56));
            Piece rook = board[rfrom];
            by_type[ROOK] ^= (1ULL << rfrom) | (1ULL << rto);
            by_color[color_of(rook)] ^= (1ULL << rfrom) | (1ULL << rto);
            occ ^= (1ULL << rfrom) | (1ULL << rto);
            state_key ^= zobrist::psq[rook][rfrom] ^ zobrist::psq[rook][rto];
            board[rfrom] = NO_PIECE;
            board[rto] = rook;
        } else if (flags == QUEEN_CASTLE) {
            Square rfrom = Square(3 + (stm * 56));
            Square rto = Square(0 + (stm * 56));
            Piece rook = board[rfrom];
            by_type[ROOK] ^= (1ULL << rfrom) | (1ULL << rto);
            by_color[color_of(rook)] ^= (1ULL << rfrom) | (1ULL << rto);
            occ ^= (1ULL << rfrom) | (1ULL << rto);
            state_key ^= zobrist::psq[rook][rfrom] ^ zobrist::psq[rook][rto];
            board[rfrom] = NO_PIECE;
            board[rto] = rook;
        }
    }

    if (flags & CAPTURE) {
        Square cap_sq = (flags == EP_CAPTURE) ? Square(to ^ 8) : to;
        Piece cap = Piece(he.captured);
        by_type[type_of(cap)] |= 1ULL << cap_sq;
        occ |= 1ULL << cap_sq;
        by_color[color_of(cap)] |= 1ULL << cap_sq;
        board[cap_sq] = cap;
    }

    castling_rights = he.castling;
    ep = he.ep;
    fifty = he.fifty;
}
