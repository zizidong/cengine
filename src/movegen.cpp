#include "movegen.h"

#include "attack.h"
#include "bitboard.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "Position.h"
#include "Move.h"
#include "PieceType.h"

template <GenType T>
void generate(const Board& b, MoveList<T>& list) {
    Color us = b.side_to_move();
    Color them = ~us;
    Bitboard occ = b.occupancy();
    Bitboard empty = ~occ;
    Bitboard enemy = b.pieces(them);
    int up = us == WHITE ? 8 : -8;
    Bitboard pawns = b.pieces(us, PAWN);
    while (pawns) {
        Square from = Square(__builtin_ctzll(pawns));
        pawns &= pawns - 1;
        Bitboard bb = 1ULL << from;

        Bitboard push = (us == WHITE ? (bb << 8) : (bb >> 8)) & empty;
        if (push) {
            int to = from + up;
            if ((to / 8) == 0 || (to / 8) == 7) {
                if (T == ALL || T == QUIETS) {
                    list.add(Move(from, Square(to), QUEEN_PROMO));
                    list.add(Move(from, Square(to), ROOK_PROMO));
                    list.add(Move(from, Square(to), BISHOP_PROMO));
                    list.add(Move(from, Square(to), KNIGHT_PROMO));
                }
            } else if (T == ALL || T == QUIETS) {
                list.add(Move(from, Square(to), QUIET));
                int dbl_to = from + 2 * up;
                if ((from / 8 == 1 && us == WHITE) || (from / 8 == 6 && us == BLACK)) {
                    if (empty & (1ULL << dbl_to))
                        list.add(Move(from, Square(dbl_to), DOUBLE_PAWN));
                }
            }
        }

        Bitboard attacks = bb::pawn_attacks[us][from] & enemy;
        while (attacks) {
            Square to = Square(__builtin_ctzll(attacks));
            attacks &= attacks - 1;
            if ((to / 8) == 0 || (to / 8) == 7) {
                if (T == ALL || T == CAPTURES) {
                    list.add(Move(from, to, QUEEN_PROMO_CAP));
                    list.add(Move(from, to, ROOK_PROMO_CAP));
                    list.add(Move(from, to, BISHOP_PROMO_CAP));
                    list.add(Move(from, to, KNIGHT_PROMO_CAP));
                }
            } else if (T == ALL || T == CAPTURES) {
                list.add(Move(from, to, CAPTURE));
            }
        }

        if (b.ep_square() != SQ_NONE) {
            Bitboard epbb = 1ULL << b.ep_square();
            if (bb::pawn_attacks[us][from] & epbb) {
                if (T == ALL || T == CAPTURES)
                    list.add(Move(from, b.ep_square(), EP_CAPTURE));
            }
        }
    }

    for (PieceType pt = KNIGHT; pt <= QUEEN; pt = PieceType(pt + 1)) {
        Bitboard pieces = b.pieces(us, pt);
        while (pieces) {
            Square from = Square(__builtin_ctzll(pieces));
            pieces &= pieces - 1;
            Bitboard attacks = bb::pseudo_attacks(pt, from, occ) & ~b.pieces(us);
            while (attacks) {
                Square to = Square(__builtin_ctzll(attacks));
                attacks &= attacks - 1;
                int flags = (enemy >> to) & 1 ? CAPTURE : QUIET;
                if (T == ALL || (T == CAPTURES && (flags & CAPTURE)) || (T == QUIETS && !(flags & CAPTURE)))
                    list.add(Move(from, to, flags));
            }
        }
    }

    Square ksq = b.king_square(us);
    Bitboard katt = bb::king_attacks[ksq] & ~b.pieces(us);
    while (katt) {
        Square to = Square(__builtin_ctzll(katt));
        katt &= katt - 1;
        int flags = (enemy >> to) & 1 ? CAPTURE : QUIET;
        if (T == ALL || (T == CAPTURES && (flags & CAPTURE)) || (T == QUIETS && !(flags & CAPTURE)))
            list.add(Move(ksq, to, flags));
    }

    if (T == ALL || T == QUIETS) {
        int cr = b.castling();
        auto sq_attacked = [&](Square s) {
            return b.attackers_to(s, occ) & enemy;
        };
        if (us == WHITE) {
            if (cr & WHITE_KING_SIDE) {
                if (!(occ & 0x60ULL) && !sq_attacked(SQ_E1) && !sq_attacked(SQ_F1) && !sq_attacked(SQ_G1))
                    list.add(Move(SQ_E1, SQ_G1, KING_CASTLE));
            }
            if (cr & WHITE_QUEEN_SIDE) {
                if (!(occ & 0x0EULL) && !sq_attacked(SQ_E1) && !sq_attacked(SQ_D1) && !sq_attacked(SQ_C1))
                    list.add(Move(SQ_E1, SQ_C1, QUEEN_CASTLE));
            }
        } else {
            if (cr & BLACK_KING_SIDE) {
                if (!(occ & 0x6000000000000000ULL) && !sq_attacked(SQ_E8) && !sq_attacked(SQ_F8) && !sq_attacked(SQ_G8))
                    list.add(Move(SQ_E8, SQ_G8, KING_CASTLE));
            }
            if (cr & BLACK_QUEEN_SIDE) {
                if (!(occ & 0x0E00000000000000ULL) && !sq_attacked(SQ_E8) && !sq_attacked(SQ_D8) && !sq_attacked(SQ_C8))
                    list.add(Move(SQ_E8, SQ_C8, QUEEN_CASTLE));
            }
        }
    }
}

template void generate<ALL>(const Board&, MoveList<ALL>&);
template void generate<CAPTURES>(const Board&, MoveList<CAPTURES>&);
template void generate<QUIETS>(const Board&, MoveList<QUIETS>&);

void generate_legal(const Board& b, MoveListAll& list) {
    list.size = 0;

    char fen[128];
    b.to_fen(fen);

    libchess::Position pos = libchess::Position::from_fen(fen).value();
    auto lml = pos.legal_move_list();

    for (const libchess::Move& lm : lml) {
        Square from = Square(lm.from_square().value());
        Square to   = Square(lm.to_square().value());
        bool is_cap = pos.is_capture_move(lm);
        bool is_promo = lm.type() == libchess::Move::Type::PROMOTION ||
                        lm.type() == libchess::Move::Type::CAPTURE_PROMOTION;

        uint16_t flags;
        switch (lm.type()) {
            case libchess::Move::Type::CASTLING:
                flags = (to == SQ_C1 || to == SQ_C8) ? QUEEN_CASTLE : KING_CASTLE;
                break;
            case libchess::Move::Type::ENPASSANT:
                flags = EP_CAPTURE;
                break;
            case libchess::Move::Type::DOUBLE_PUSH:
                flags = DOUBLE_PAWN;
                break;
            case libchess::Move::Type::PROMOTION: {
                int pt = lm.promotion_piece_type()->value();
                if      (pt == 1) flags = KNIGHT_PROMO;
                else if (pt == 2) flags = BISHOP_PROMO;
                else if (pt == 3) flags = ROOK_PROMO;
                else              flags = QUEEN_PROMO;
                break;
            }
            case libchess::Move::Type::CAPTURE_PROMOTION: {
                int pt = lm.promotion_piece_type()->value();
                if      (pt == 1) flags = KNIGHT_PROMO_CAP;
                else if (pt == 2) flags = BISHOP_PROMO_CAP;
                else if (pt == 3) flags = ROOK_PROMO_CAP;
                else              flags = QUEEN_PROMO_CAP;
                break;
            }
            default:
                flags = is_cap ? CAPTURE : QUIET;
                break;
        }
        (void)is_promo;
        list.add(Move(from, to, flags));
    }
}
