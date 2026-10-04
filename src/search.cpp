#include "search.h"

#include "evaluate.h"
#include "nnue.h"
#include "tt.h"
#include "zobrist.h"
#include "sync_out.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace search {

namespace {

using clock_t = std::chrono::steady_clock;

constexpr int MAX_PLY = 128;
// Scores above this (in absolute value) are mate scores.
constexpr int MATE_IN_MAX = VALUE_MATE - MAX_PLY;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_total_nodes{0};

constexpr int LMR_MAX_DEPTH = 64;
constexpr int LMR_MAX_MOVES = 64;

int lmr_table[LMR_MAX_DEPTH][LMR_MAX_MOVES];

void init_lmr() {
    for (int d = 1; d < LMR_MAX_DEPTH; ++d)
        for (int m = 1; m < LMR_MAX_MOVES; ++m)
            lmr_table[d][m] = int(0.77 + std::log(double(d)) * std::log(double(m)) / 2.36);
}

struct Searcher {
    Board board;
    SearchLimits limits;
    clock_t::time_point start;
    bool stop_soft = false;   // per-thread; cross-thread stop is g_stop
    bool is_main = true;

    uint64_t nodes = 0;
    int seldepth = 0;         // max ply reached (negamax + qsearch) this search
    int best_score = 0;
    Move best_move;
    Move ponder_move;

    int max_time_ms  = 0;
    int opt_time_ms  = 0;

    int killer[MAX_PLY][2];
    int history[PIECE_NB][SQ_NB];
    Move counterMove[PIECE_NB][SQ_NB];

    // triangular PV
    Move pv_table[MAX_PLY][MAX_PLY];
    int  pv_len[MAX_PLY];

    // per-line extension budget (prevents check-extension explosion)
    int ext_count[MAX_PLY];
    static constexpr int MAX_EXT = 8;

    bool time_up() const {
        if (limits.infinite) return false;
        if (limits.nodes >= 0 && nodes >= uint64_t(limits.nodes)) return true;
        auto elapsed_ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - start).count());
        if (max_time_ms > 0 && elapsed_ms >= max_time_ms) return true;
        if (limits.movetime >= 0 && elapsed_ms >= max_time_ms) return true;
        return false;
    }

    bool out_of_optimum_time() const {
        if (limits.infinite || limits.depth > 0) return false;
        auto elapsed_ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - start).count());
        if (opt_time_ms > 0 && elapsed_ms >= opt_time_ms) return true;
        return false;
    }

    void check_time() {
        if (!stop_soft && (g_stop.load(std::memory_order_relaxed) || time_up())) stop_soft = true;
    }

    void compute_times() {
        max_time_ms = 0;
        opt_time_ms = 0;

        if (limits.movetime >= 0) {
            opt_time_ms = std::max(1, limits.movetime - std::min(50, limits.movetime / 20));
            max_time_ms = limits.movetime;
            return;
        }
        if (limits.wtime >= 0 && limits.btime >= 0) {
            int time = (board.side_to_move() == WHITE) ? limits.wtime : limits.btime;
            int inc  = (board.side_to_move() == WHITE) ? limits.winc  : limits.binc;
            if (time <= 0) { opt_time_ms = 1; max_time_ms = 1; return; }
            int moves_left = limits.movestogo > 0 ? limits.movestogo : 30;
            opt_time_ms = std::max(1, time / moves_left + inc * 3 / 4);
            opt_time_ms = std::min(opt_time_ms, time / 4);
            max_time_ms = std::min(time / 2, opt_time_ms * 5);
            max_time_ms = std::max(max_time_ms, opt_time_ms);
            return;
        }
        // depth/infinite/nodes only: rely on the hard cap elsewhere
    }

    int score_move(Move m, int ply, const Move& tt_move, Move prev_move) const {
        if (m == tt_move) return 1 << 24;

        if (m.is_capture()) {
            int cap = int(board.type_on(m.to()));
            int mov = int(board.type_on(m.from()));
            int v = (1 << 20) + eval::PIECE_VALUE[cap] * 16 - eval::PIECE_VALUE[mov];
            if (m.is_promotion()) v += eval::PIECE_VALUE[m.promo_type()] * 16;
            return v;
        }
        if (m.is_promotion()) return (1 << 19) + eval::PIECE_VALUE[m.promo_type()];

        if (m == killer_entry(ply, 0)) return (1 << 18);
        if (m == killer_entry(ply, 1)) return (1 << 17);

        if (m == counterMove[prev_move.from()][prev_move.to()] && prev_move != Move::none())
            return (1 << 16);

        return history[move_piece_index(m)][m.to()];
    }

    Move killer_entry(int ply, int slot) const { return slot == 0 ? Move(killer[ply][0]) : Move(killer[ply][1]); }
    int move_piece_index(Move m) const { return board.piece_on(m.from()); }

    // Move ordering: score every move once, then selection-sort on the cached
    // scores (score_move is comparatively expensive and was re-run per pick).
    void score_all(MoveListAll& list, int ply, const Move& tt_move, Move prev_move) const {
        for (int i = 0; i < list.size; ++i)
            list.scores[i] = score_move(list.moves[i], ply, tt_move, prev_move);
    }
    void score_all_q(MoveListCap& list) const {
        for (int i = 0; i < list.size; ++i) list.scores[i] = mvv_lva(list.moves[i]);
    }

    void pick_next(MoveListAll& list, int i) const {
        int best = i, bs = list.scores[i];
        for (int j = i + 1; j < list.size; ++j)
            if (list.scores[j] > bs) { bs = list.scores[j]; best = j; }
        if (best != i) {
            std::swap(list.moves[i], list.moves[best]);
            std::swap(list.scores[i], list.scores[best]);
        }
    }

    void pick_next_q(MoveListCap& list, int i) const {
        int best = i, bs = list.scores[i];
        for (int j = i + 1; j < list.size; ++j)
            if (list.scores[j] > bs) { bs = list.scores[j]; best = j; }
        if (best != i) {
            std::swap(list.moves[i], list.moves[best]);
            std::swap(list.scores[i], list.scores[best]);
        }
    }

    int mvv_lva(Move m) const {
        auto sv = [](int t) { return eval::PIECE_VALUE[t < PIECE_TYPE_NB ? t : 0]; };
        int v = sv(int(board.type_on(m.to()))) * 16;
        v += sv(int(m.promo_type())) * 16;
        v -= sv(int(board.type_on(m.from())));
        return v;
    }

    void make_null_move() {
        HistoryEntry& he = board.history[board.history_ply++];
        he.move = Move::none();
        he.captured = NO_PIECE;
        he.castling = uint8_t(board.castling_rights);
        he.ep = board.ep;
        he.fifty = uint8_t(board.fifty);
        he.key = board.state_key;

        board.ep = SQ_NONE;
        board.fifty++;
        board.ply++;
        board.stm = ~board.stm;
        board.state_key ^= zobrist::side;
    }

    void unmake_null_move() {
        HistoryEntry& he = board.history[--board.history_ply];
        board.stm = ~board.stm;
        board.ply--;
        board.fifty = he.fifty;
        board.ep = he.ep;
        board.state_key = he.key;
    }

    // In-check quiescence: stand-pat is meaningless and every legal evasion
    // (including quiet ones) must be tried, otherwise qsearch can return a
    // bogus score or miss a mate.
    int quiescence_evasions(int alpha, int beta, int ply) {
        if (ply > seldepth) seldepth = ply;
        MoveListAll list;
        generate(board, list);
        score_all(list, ply, Move::none(), Move::none());

        int best  = -VALUE_INFINITE;
        int legal = 0;
        for (int i = 0; i < list.size; ++i) {
            pick_next(list, i);
            Move m = list.moves[i];

            board.make_move(m);
            if (board.moved_into_check()) { board.unmake_move(); continue; }
            ++legal;
            eval::nnue_on_make(board, ply <= eval::nnue_big_net_ply());
            int score = -quiescence(-beta, -alpha, ply + 1);
            board.unmake_move();
            eval::nnue_on_unmake();

            if (stop_soft) return 0;
            if (score > best) best = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
        }
        if (legal == 0) return -VALUE_MATE + ply;
        return best;
    }

    int quiescence(int alpha, int beta, int ply) {
        ++nodes;
        if (ply > seldepth) seldepth = ply;
        if ((nodes & 1023) == 0) check_time();
        if (stop_soft) return 0;
        if (ply >= MAX_PLY - 1) return eval::evaluate(board);

        if (board.checkers() != 0) return quiescence_evasions(alpha, beta, ply);

        int stand_pat = eval::evaluate(board);
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;

        MoveListCap list;
        generate(board, list);
        score_all_q(list);
        const Bitboard pinned = board.pinned_pieces(board.side_to_move());

        int best = stand_pat;
        for (int i = 0; i < list.size; ++i) {
            pick_next_q(list, i);
            Move m = list.moves[i];

            // delta pruning
            if (stand_pat + eval::PIECE_VALUE[board.type_on(m.to())] + 200 < alpha && !m.is_promotion())
                continue;

            // prune losing captures at shallow quiescence depth
            if (!board.see_ge(m, 0)) continue;

            if (!board.legal_precheck(m, pinned)) continue;
            board.make_move(m);
            if (m.is_ep() && board.moved_into_check()) { board.unmake_move(); continue; }
            eval::nnue_on_make(board, ply <= eval::nnue_big_net_ply());
            int score = -quiescence(-beta, -alpha, ply + 1);
            board.unmake_move();
            eval::nnue_on_unmake();

            if (stop_soft) return 0;
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
            if (score > best) best = score;
        }
        return alpha;
    }

    int negamax(int alpha, int beta, int depth, int ply, bool pv_node, Move prev_move) {
        pv_len[ply] = 0;

        if (depth <= 0) return quiescence(alpha, beta, ply);

        ++nodes;
        if (ply > seldepth) seldepth = ply;
        if ((nodes & 1023) == 0) check_time();
        if (stop_soft) return 0;
        if (ply >= MAX_PLY - 1) return eval::evaluate(board);

        bool in_check = board.checkers() != 0;

        // draw detection
        if (ply > 0 && (board.fifty >= 100 || board.is_repetition()
                        || board.is_insufficient_material()))
            return 0;

        // mate distance pruning
        if (!pv_node) {
            alpha = std::max(alpha, -VALUE_MATE + ply);
            beta  = std::min(beta,  VALUE_MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }

        // (check extension is applied by the parent with a per-line budget)

        // TT probe
        bool found;
        TTEntry* te = TT.probe(board.key(), found);
        Move tt_move = found ? Move(te->move16) : Move::none();

        if (found && te->depth >= depth && tt_bound(te) != TT_NONE &&
            (!pv_node || tt_bound(te) == TT_EXACT)) {
            int s = te->score;
            if (s > MATE_IN_MAX) s -= ply;
            else if (s < -MATE_IN_MAX) s += ply;
            uint8_t b = tt_bound(te);
            if (b == TT_EXACT) return s;
            if (b == TT_LOWER && s >= beta) return s;
            if (b == TT_UPPER && s <= alpha) return s;
        }

        int static_eval;
        if (in_check)
            static_eval = VALUE_NONE;
        else if (found && te->eval != VALUE_NONE)
            static_eval = te->eval;               // reuse cached static eval
        else
            static_eval = eval::evaluate(board);

        // reverse futility pruning
        if (!pv_node && !in_check && depth <= 7 && static_eval != VALUE_NONE
            && static_eval - 85 * depth >= beta)
            return static_eval;

        // null-move pruning
        bool has_nonpawn = (board.pieces(board.side_to_move())
                            & ~board.pieces(board.side_to_move(), PAWN)
                            & ~board.pieces(board.side_to_move(), KING)) != 0;
        if (!pv_node && !in_check && depth >= 3 && has_nonpawn && static_eval >= beta) {
            make_null_move();
            eval::nnue_on_make(board, (ply + 1) <= eval::nnue_big_net_ply());
            int R = 3 + depth / 4;
            int s = -negamax(-beta, -beta + 1, depth - R, ply + 1, false, Move::none());
            eval::nnue_on_unmake();
            unmake_null_move();
            if (stop_soft) return 0;
            if (s >= beta) return beta;
        }

        // Internal iterative reduction: with no hash move at a deep node, search
        // one ply shallower first; the children usually supply a hash move.
        if (depth >= 4 && tt_move == Move::none() && !in_check) --depth;

        MoveListAll list;
        generate(board, list);
        score_all(list, ply, tt_move, prev_move);
        const Bitboard pinned = board.pinned_pieces(board.side_to_move());
        const bool     inCheck = in_check;

        int best = -VALUE_INFINITE;
        Move best_move_local = Move::none();
        int flag = TT_UPPER;
        int moves_searched = 0;
        int legal_count = 0;

        for (int i = 0; i < list.size; ++i) {
            pick_next(list, i);
            Move m = list.moves[i];

            bool is_quiet = m.is_quiet();
            bool is_cap   = m.is_capture();

            // SEE is evaluated on the pre-move position.
            bool see_ok = true;
            if (is_cap && !m.is_promotion() && depth <= 5 && !pv_node)
                see_ok = board.see_ge(m, -60 * depth);

            if (!inCheck && !board.legal_precheck(m, pinned)) continue;
            board.make_move(m);
            if ((inCheck || m.is_ep()) && board.moved_into_check()) { board.unmake_move(); continue; }
            ++legal_count;
            eval::nnue_on_make(board, (eval::nnue_big_net_pv() && pv_node) || (ply + 1) <= eval::nnue_big_net_ply());
            bool gives_check = board.checkers() != 0;

            // Pruning (never prune the first searched move).
            if (moves_searched > 0 && !pv_node && !in_check) {
                bool prune = false;
                if (is_quiet && depth <= 4 && moves_searched >= 4 + depth * depth)
                    prune = true;                                   // late move pruning
                else if (is_quiet && depth <= 6 && static_eval != VALUE_NONE
                         && static_eval + 100 + 120 * depth <= alpha)
                    prune = true;                                   // futility pruning
                else if (is_cap && !see_ok)
                    prune = true;                                   // SEE pruning
                if (prune) {
                    eval::nnue_on_unmake();
                    board.unmake_move();
                    continue;
                }
            }

            int score;
            int ext = (gives_check && ext_count[ply] < MAX_EXT) ? 1 : 0;
            ext_count[ply + 1] = ext_count[ply] + ext;
            if (moves_searched == 0) {
                score = -negamax(-beta, -alpha, depth - 1 + ext, ply + 1, pv_node, m);
            } else {
                int R = 0;
                if (is_quiet && !gives_check && !in_check && depth >= 3 && moves_searched >= 2) {
                    R = lmr_table[std::min(depth, LMR_MAX_DEPTH - 1)]
                                 [std::min(moves_searched, LMR_MAX_MOVES - 1)];
                    if (pv_node) R -= 1;
                    R = std::max(0, R);
                }
                score = -negamax(-alpha - 1, -alpha, depth - 1 - R + ext, ply + 1, false, m);
                if (score > alpha && R > 0)
                    score = -negamax(-alpha - 1, -alpha, depth - 1 + ext, ply + 1, false, m);
                if (score > alpha && score < beta)
                    score = -negamax(-beta, -alpha, depth - 1 + ext, ply + 1, pv_node, m);
            }

            board.unmake_move();
            eval::nnue_on_unmake();
            ++moves_searched;

            if (stop_soft) return 0;

            if (score > best) {
                best = score;
                best_move_local = m;

                if (score > alpha) {
                    alpha = score;
                    flag = TT_EXACT;

                    // update PV
                    pv_table[ply][0] = m;
                    int child_len = pv_len[ply + 1];
                    for (int k = 0; k < child_len && k < MAX_PLY - 1; ++k)
                        pv_table[ply][k + 1] = pv_table[ply + 1][k];
                    pv_len[ply] = std::min(child_len + 1, MAX_PLY - 1);

                    if (score >= beta) {
                        flag = TT_LOWER;
                        if (is_quiet) {
                            update_quiet_stats(m, depth, ply, prev_move);
                        }
                        break;
                    }
                }
            }
        }

        if (legal_count == 0)
            return in_check ? -VALUE_MATE + ply : 0;
        if (moves_searched == 0)
            return (static_eval != VALUE_NONE) ? static_eval : alpha;

        int store_score = best;
        if (store_score > MATE_IN_MAX) store_score += ply;
        else if (store_score < -MATE_IN_MAX) store_score -= ply;
        int store_eval = (static_eval == VALUE_NONE) ? VALUE_NONE : static_eval;
        TT.store(board.key(), best_move_local, store_score, store_eval, depth, flag);

        return best;
    }

    void update_quiet_stats(Move m, int depth, int ply, Move prev_move) {
        if (uint32_t(killer[ply][0]) != m.raw()) {
            killer[ply][1] = killer[ply][0];
            killer[ply][0] = m.raw();
        }
        int& h = history[move_piece_index(m)][m.to()];
        h += depth * depth;
        if (h > (1 << 28)) {
            // rescale
            for (int p = 0; p < PIECE_NB; ++p)
                for (int s = 0; s < SQ_NB; ++s)
                    history[p][s] /= 2;
        }
        if (prev_move != Move::none())
            counterMove[prev_move.from()][prev_move.to()] = m;
    }

    std::string move_to_uci(Move m) const {
        char buf[6];
        buf[0] = char('a' + FILE_OF(m.from()));
        buf[1] = char('1' + RANK_OF(m.from()));
        buf[2] = char('a' + FILE_OF(m.to()));
        buf[3] = char('1' + RANK_OF(m.to()));
        if (m.is_promotion()) {
            char pc = m.promo_type() == QUEEN ? 'q' : m.promo_type() == ROOK ? 'r' : m.promo_type() == BISHOP ? 'b' : 'n';
            buf[4] = pc;
            buf[5] = '\0';
        } else {
            buf[4] = '\0';
        }
        return std::string(buf);
    }

    void report(int depth, int score, uint64_t n, int time_ms, Move root_move) {
        // Build the PV by walking the TT and validating every move against the
        // legal move list, so the reported line is always a legal sequence.
        Move pv[MAX_PLY];
        int pv_n = 0;
        {
            Board b = board;
            for (int i = 0; i < depth && i < MAX_PLY - 1; ++i) {
                Move m;
                if (i == 0) {
                    m = root_move;
                } else {
                    bool found;
                    TTEntry* te = TT.probe(b.key(), found);
                    if (!found) break;
                    m = Move(te->move16);
                }
                if (m == Move::none()) break;

                MoveListAll list;
                generate(b, list);
                Move chosen = Move::none();
                for (int j = 0; j < list.size; ++j) {
                    if (list.moves[j].raw() != m.raw()) continue;
                    b.make_move(list.moves[j]);
                    if (!b.moved_into_check()) chosen = list.moves[j];
                    b.unmake_move();
                    break;
                }
                if (chosen == Move::none()) break;
                b.make_move(chosen);
                pv[pv_n++] = chosen;
            }
        }

        std::string out = "info depth " + std::to_string(depth) + " seldepth "
                        + std::to_string(seldepth) + " ";
        if (score > MATE_IN_MAX) {
            int mate_in = (VALUE_MATE - score + 1) / 2;
            out += "score mate " + std::to_string(mate_in) + " ";
        } else if (score < -MATE_IN_MAX) {
            int mate_in = (VALUE_MATE + score + 1) / 2;
            out += "score mate -" + std::to_string(mate_in) + " ";
        } else {
            out += "score cp " + std::to_string(score) + " ";
        }
        out += "nodes " + std::to_string(n) + " ";
        out += "nps " + std::to_string(time_ms > 0 ? (n * 1000 / time_ms) : n) + " ";
        out += "time " + std::to_string(time_ms) + " ";
        out += "pv";
        for (int i = 0; i < pv_n; ++i) {
            out += " ";
            out += move_to_uci(pv[i]);
        }
        out += "\n";
        sync_out::print(out);
    }
};

Board g_root_board;
std::vector<Searcher> g_searchers;
int g_threads = 1;

void run_search(Searcher& s, int thread_id, SearchLimits limits, SearchResult* result) {
    s.limits = limits;
    s.start = clock_t::now();
    s.nodes = 0;
    s.seldepth = 0;
    s.best_move = Move::none();
    s.ponder_move = Move::none();
    s.best_score = 0;
    s.stop_soft = false;
    s.is_main = (thread_id == 0);

    s.board = g_root_board;
    eval::nnue_set_root(s.board);
    s.compute_times();

    // Terminal root position (checkmate or stalemate): report the result
    // instead of returning a bare "bestmove 0000".
    {
        MoveListAll list;
        generate(s.board, list);
        const Bitboard pinned = s.board.pinned_pieces(s.board.side_to_move());
        const bool     inCheck = s.board.checkers() != 0;
        bool has_legal = false;
        for (int i = 0; i < list.size && !has_legal; ++i) {
            const Move m = list.moves[i];
            if (!inCheck && !s.board.legal_precheck(m, pinned)) continue;
            s.board.make_move(m);
            has_legal = !s.board.moved_into_check();
            s.board.unmake_move();
        }
        if (!has_legal) {
            if (result) {
                result->best = Move::none();
                result->score = inCheck ? -VALUE_MATE : 0;
                result->depth = 0;
                result->nodes = 0;
            }
            if (s.is_main)
                sync_out::print("info depth 0 seldepth 0 score ",
                                inCheck ? "mate 0" : "cp 0", " nodes 0 nps 0 time 0 pv\n");
            return;
        }
    }

    int max_depth = limits.depth > 0 ? limits.depth : MAX_PLY - 1;

    Move prev_best = Move::none();
    int prev_score = 0;
    s.ext_count[0] = 0;

    // Lazy SMP: helper threads start at staggered depths to reduce redundant
    // shallow searching while still sharing the transposition table.
    const int first_depth = (thread_id == 0) ? 1 : std::min(1 + thread_id, 8);

    for (int depth = first_depth; depth <= max_depth; ++depth) {
        int window_alpha = -VALUE_INFINITE;
        int window_beta  =  VALUE_INFINITE;
        int asp_delta = 16;
        if (depth >= 5) {
            window_alpha = std::max(int(-VALUE_INFINITE), prev_score - asp_delta);
            window_beta  = std::min(int(VALUE_INFINITE), prev_score + asp_delta);
        }

        int best_score = -VALUE_INFINITE;
        Move best_move = Move::none();
        bool completed = true;

        while (true) {
            best_score = -VALUE_INFINITE;
            best_move = Move::none();

            MoveListAll list;
            generate(s.board, list);
            s.score_all(list, 0, prev_best, Move::none());
            const Bitboard pinned = s.board.pinned_pieces(s.board.side_to_move());
            const bool     inCheck = s.board.checkers() != 0;
            s.pv_len[0] = 0;

            int a = window_alpha;  // running alpha for the root move loop

            for (int i = 0; i < list.size; ++i) {
                s.pick_next(list, i);
                Move m = list.moves[i];

                if (!inCheck && !s.board.legal_precheck(m, pinned)) continue;
                s.board.make_move(m);
                if ((inCheck || m.is_ep()) && s.board.moved_into_check()) { s.board.unmake_move(); continue; }
                eval::nnue_on_make(s.board, eval::nnue_big_net_ply() >= 1);

                s.ext_count[1] = s.ext_count[0]
                    + ((s.board.checkers() != 0 && s.ext_count[0] < Searcher::MAX_EXT) ? 1 : 0);

                int score;
                if (i == 0) {
                    score = -s.negamax(-window_beta, -a, depth - 1, 1, true, m);
                } else {
                    score = -s.negamax(-a - 1, -a, depth - 1, 1, false, m);
                    if (score > a && score < window_beta)
                        score = -s.negamax(-window_beta, -a, depth - 1, 1, true, m);
                }
                int child_len = s.pv_len[1];
                s.board.unmake_move();
                eval::nnue_on_unmake();

                if (s.stop_soft) { completed = false; break; }

                if (score > best_score) {
                    best_score = score;
                    best_move = m;
                    s.pv_table[0][0] = m;
                    for (int k = 0; k < child_len; ++k)
                        s.pv_table[0][k + 1] = s.pv_table[1][k];
                    s.pv_len[0] = std::min(child_len + 1, MAX_PLY - 1);
                    if (score > a) a = score;
                }
            }

            if (!completed) break;

            // aspiration window re-search against the *original* window bounds
            if (best_score <= window_alpha && window_alpha > -VALUE_INFINITE) {
                asp_delta += asp_delta / 2 + 5;
                window_alpha = std::max(int(-VALUE_INFINITE), best_score - asp_delta);
                window_beta  = std::min(int(VALUE_INFINITE),  best_score + asp_delta);
                continue;
            }
            if (best_score >= window_beta && window_beta < VALUE_INFINITE) {
                asp_delta += asp_delta / 2 + 5;
                window_beta = std::min(int(VALUE_INFINITE), best_score + asp_delta);
                continue;
            }
            break;
        }

        if (!completed && prev_best != Move::none())
            break;
        if (best_move == Move::none())
            break;

        prev_best = best_move;
        prev_score = best_score;
        s.best_move = best_move;
        s.best_score = best_score;

        if (s.is_main) {
            int elapsed_ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - s.start).count());
            s.report(depth, best_score, s.nodes, elapsed_ms, best_move);
        }

        if (result) {
            result->best = best_move;
            result->score = best_score;
            result->depth = depth;
            result->nodes = s.nodes;
        }

        if (s.stop_soft) break;
        if (limits.depth > 0 && depth >= limits.depth) break;
        if (s.out_of_optimum_time()) break;
    }

    // Fallback: if stopped before any move was completed, return any legal move.
    if (result && result->best == Move::none()) {
        MoveListAll list;
        generate(s.board, list);
        for (int i = 0; i < list.size; ++i) {
            s.board.make_move(list.moves[i]);
            bool ok = !s.board.moved_into_check();
            s.board.unmake_move();
            if (ok) {
                result->best = list.moves[i];
                s.best_move = list.moves[i];
                break;
            }
        }
    }

    g_total_nodes.fetch_add(s.nodes, std::memory_order_relaxed);
}

}  // namespace

bool stopped = false;

void init() {
    eval::init();
    init_lmr();
}

void reset() {
    for (Searcher& s : g_searchers) {
        std::memset(s.killer, 0, sizeof(s.killer));
        std::memset(s.history, 0, sizeof(s.history));
        for (int p = 0; p < PIECE_NB; ++p)
            for (int sq = 0; sq < SQ_NB; ++sq)
                s.counterMove[p][sq] = Move::none();
    }
    TT.clear_entries();
}

void set_position(const Board& b) {
    g_root_board = b;
}

void set_threads(int n) {
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    g_threads = n;
}

void stop() { stopped = true; g_stop.store(true); }
void clear_stop() { g_stop.store(false); }

void bench(int depth) {
    static const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        "8/8/8/8/5k2/8/6p1/6K1 w - - 0 1",
    };

    uint64_t total_nodes = 0;
    nnue_full::AccStats astats{0, 0, 0, 0};
    uint64_t cyc_build = 0, cyc_apply = 0, cyc_finish = 0;
    uint64_t cyc_bapp = 0, cyc_bsort = 0;
    auto t0 = clock_t::now();
    uint64_t c0 = nnue::tsc_now();
    TT.clear_entries();
    for (const char* fen : fens) {
        Board b;
        b.set_fen(fen);
        set_position(b);
        SearchLimits lim;
        lim.depth = depth;
        SearchResult r;
        clear_stop();
        go(lim, r);
        total_nodes += g_total_nodes.load(std::memory_order_relaxed);
        nnue_full::AccStats s = nnue::acc_stats();
        astats.incremental_evals += s.incremental_evals;
        astats.full_evals       += s.full_evals;
        astats.pushes           += s.pushes;
        astats.diff_applies     += s.diff_applies;
        astats.big_evals        += s.big_evals;
        uint64_t b2, a2, f2;
        nnue::acc_cycles(b2, a2, f2);
        cyc_build += b2; cyc_apply += a2; cyc_finish += f2;
        uint64_t ap2, so2;
        nnue::acc_build_cycles(ap2, so2);
        cyc_bapp += ap2; cyc_bsort += so2;
    }
    uint64_t total_cyc = nnue::tsc_now() - c0;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - t0).count();
    sync_out::print("bench depth ", depth, " nodes ", total_nodes, " time ", ms,
                    " nps ", (ms > 0 ? total_nodes * 1000 / uint64_t(ms) : total_nodes), "\n");
    uint64_t evals = astats.incremental_evals + astats.full_evals;
    double hit = evals ? (100.0 * double(astats.incremental_evals) / double(evals)) : 0.0;
    double per_push = astats.pushes ? (double(astats.diff_applies) / double(astats.pushes)) : 0.0;
    sync_out::print("nnue: inc_evals ", astats.incremental_evals,
                    " full_evals ", astats.full_evals,
                    " incremental_hit ", hit, "%",
                    " pushes ", astats.pushes,
                    " diff_applies/push ", per_push, " big_evals ", astats.big_evals, "\n");
    double tc = double(total_cyc ? total_cyc : 1);
    sync_out::print("cyc: total ", total_cyc,
                    " build_features ", cyc_build, " (", 100.0 * double(cyc_build) / tc, "%)",
                    " apply_diff ", cyc_apply, " (", 100.0 * double(cyc_apply) / tc, "%)",
                    " finish ", cyc_finish, " (", 100.0 * double(cyc_finish) / tc, "%)",
                    " nnue_total ", 100.0 * double(cyc_build + cyc_apply + cyc_finish) / tc,
                    "%\n");
    sync_out::print("cyc/eval build ", evals ? double(cyc_build) / double(evals) : 0.0,
                    " apply ", evals ? double(cyc_apply) / double(evals) : 0.0,
                    " finish ", evals ? double(cyc_finish) / double(evals) : 0.0,
                    " total ", evals ? double(cyc_build + cyc_apply + cyc_finish) / double(evals)
                                     : 0.0,
                    "\n");
    sync_out::print("cyc/build: append ", cyc_bapp, " (", 100.0 * double(cyc_bapp) / tc, "%)",
                    " sort ", cyc_bsort, " (", 100.0 * double(cyc_bsort) / tc, "%)\n");
}

void go(SearchLimits limits, SearchResult& result) {
    g_stop.store(false);
    stopped = false;
    result = SearchResult{};

    TT.new_search();
    g_total_nodes.store(0, std::memory_order_relaxed);

    if (int(g_searchers.size()) < g_threads)
        g_searchers.resize(g_threads);

    std::vector<std::thread> workers;
    workers.reserve(g_threads > 0 ? g_threads - 1 : 0);
    for (int i = 1; i < g_threads; ++i)
        workers.emplace_back([i, limits]() { run_search(g_searchers[i], i, limits, nullptr); });

    run_search(g_searchers[0], 0, limits, &result);

    // Ask helper threads to stop and wait for them.
    g_stop.store(true);
    for (auto& w : workers)
        if (w.joinable()) w.join();
}

}  // namespace search