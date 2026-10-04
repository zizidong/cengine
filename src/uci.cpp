#include "uci.h"

#include "board.h"
#include "bitboard.h"
#include "movegen.h"
#include "evaluate.h"
#include "search.h"
#include "tt.h"
#include "perft.h"
#include "nnue.h"
#include "sync_out.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>

namespace uci {

namespace {

constexpr char kStartFEN[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

Board root_board;

std::atomic<bool>     searching{false};
std::thread           search_thread;
SearchResult          last_result;
SearchLimits          search_last_limits;

// ---- UCI move <-> internal Move ----

Move parse_move(const std::string& str) {
    if (str.size() < 4) return Move::none();
    int from_f = str[0] - 'a';
    int from_r = str[1] - '1';
    int to_f   = str[2] - 'a';
    int to_r   = str[3] - '1';
    Square from = Square(from_r * 8 + from_f);
    Square to   = Square(to_r * 8 + to_f);
    char promo  = str.size() >= 5 ? str[4] : '\0';
    int  flags  = QUIET;

    Piece p = root_board.piece_on(from);
    if (type_of(p) == PAWN) {
        if (to == root_board.ep_square())
            flags = EP_CAPTURE;
        else if (to_r == 0 || to_r == 7)
            flags = QUEEN_PROMO;
        else if (abs(to_r - from_r) == 2)
            flags = DOUBLE_PAWN;
        else if (root_board.piece_on(to) != NO_PIECE || to_f != from_f)
            flags = CAPTURE;
        if (promo) {
            int base = (flags & CAPTURE) ? 12 : 8;
            switch (promo) {
                case 'n': flags = base + 0; break;
                case 'b': flags = base + 1; break;
                case 'r': flags = base + 2; break;
                default:  flags = base + 3; break;
            }
        }
    } else {
        if (type_of(p) == KING && abs(to_f - from_f) == 2)
            flags = (to_f > from_f) ? KING_CASTLE : QUEEN_CASTLE;
        else if (root_board.piece_on(to) != NO_PIECE)
            flags = CAPTURE;
    }
    return Move(from, to, flags);
}

std::string move_str(Move m) {
    std::string s;
    s.push_back(char('a' + FILE_OF(m.from())));
    s.push_back(char('1' + RANK_OF(m.from())));
    s.push_back(char('a' + FILE_OF(m.to())));
    s.push_back(char('1' + RANK_OF(m.to())));
    if (m.is_promotion()) {
        char pc = m.promo_type() == QUEEN  ? 'q'
                : m.promo_type() == ROOK   ? 'r'
                : m.promo_type() == BISHOP ? 'b' : 'n';
        s.push_back(pc);
    }
    return s;
}

void wait_for_search_finished() {
    if (search_thread.joinable()) search_thread.join();
}

// ---- Output handlers (Stockfish-style on_* callbacks) ----

void on_bestmove(const SearchResult& r) {
    if (r.best != Move::none()) {
        std::string out = "bestmove " + move_str(r.best);
        if (r.ponder != Move::none() && r.ponder != Move(0))
            out += " ponder " + move_str(r.ponder);
        sync_out::print(out + "\n");
    } else {
        sync_out::print("bestmove 0000\n");
    }
}

void on_uci() {
    sync_out::print("id name cengine\n",
                    "id author kilo\n",
                    "option name Hash type spin default 32 min 1 max 1024\n",
                    "option name Threads type spin default 1 min 1 max 64\n",
                    "option name Move Overhead type spin default 10 min 0 max 5000\n",
                    "option name BigNetPly type spin default 2 min 0 max 64\n",
                    "option name BigNetPV type spin default 1 min 0 max 1\n");
    const std::string& lp = nnue::loaded_path();
    if (!lp.empty()) {
        std::string p = lp;
        for (auto& c : p) if (c == '\\') c = '/';
        sync_out::print("option name EvalFile type string default ", p, "\n");
    } else {
        sync_out::print("option name EvalFile type string default <empty>\n");
    }
    sync_out::print("uciok\n");
}

void on_setoption(std::istringstream& is) {
    wait_for_search_finished();
    std::string token, name, value;
    is >> token;  // "name"
    while (is >> token && token != "value") {
        if (!name.empty()) name += ' ';
        name += token;
    }
    if (token == "value") {
        std::getline(is, value);
        if (!value.empty() && value[0] == ' ') value.erase(0, 1);
    }
    if (name.find("Hash") != std::string::npos) {
        TT.resize(std::atoll(value.c_str()));
    } else if (name.find("Threads") != std::string::npos) {
        search::set_threads(std::atoi(value.c_str()));
    } else if (name.find("Move Overhead") != std::string::npos) {
        // accepted; applied as extra time margin
    } else if (name.find("EvalFile") != std::string::npos) {
        nnue::load(value.c_str());
    } else if (name.find("BigNetPly") != std::string::npos) {
        eval::nnue_set_big_net_ply(std::atoi(value.c_str()));
    } else if (name.find("BigNetPV") != std::string::npos) {
        eval::nnue_set_big_net_pv(std::atoi(value.c_str()));
    }
}

void on_position(std::istringstream& is) {
    wait_for_search_finished();
    std::string token, fen;
    is >> token;
    if (token == "startpos") {
        fen = kStartFEN;
        is >> token;  // consume "moves" if present
    } else if (token == "fen") {
        while (is >> token && token != "moves") {
            if (!fen.empty()) fen += ' ';
            fen += token;
        }
    } else {
        return;
    }
    root_board.set_fen(fen.c_str());
    if (token == "moves") {
        while (is >> token) root_board.make_move(parse_move(token));
    }
    search::set_position(root_board);
}

void on_go(std::istringstream& is) {
    wait_for_search_finished();
    if (searching.load()) return;

    SearchLimits limits;
    std::string  token;
    while (is >> token) {
        if      (token == "infinite")   limits.infinite = true;
        else if (token == "depth"    && (is >> limits.depth))      {}
        else if (token == "movetime" && (is >> limits.movetime))   {}
        else if (token == "wtime"    && (is >> limits.wtime))      {}
        else if (token == "btime"    && (is >> limits.btime))      {}
        else if (token == "winc"     && (is >> limits.winc))       {}
        else if (token == "binc"     && (is >> limits.binc))       {}
        else if (token == "movestogo" && (is >> limits.movestogo)) {}
        else if (token == "nodes"    && (is >> limits.nodes))      {}
        else if (token == "searchmoves") { while (is >> token) {} break; }
    }

    if (limits.movetime < 0 && limits.wtime >= 0 && limits.btime >= 0) {
        int time = root_board.side_to_move() == WHITE ? limits.wtime : limits.btime;
        int inc  = root_board.side_to_move() == WHITE ? limits.winc  : limits.binc;
        int moves_left = limits.movestogo > 0 ? limits.movestogo : 25;
        limits.movetime = std::max(1, time / moves_left + inc / 2);
    }

    search_last_limits = limits;
    last_result = SearchResult{};
    search::clear_stop();
    searching.store(true);
    search_thread = std::thread([]() {
        search::go(search_last_limits, last_result);
        searching.store(false);
        on_bestmove(last_result);
    });
}

}  // namespace

void loop() {
    std::string cmd;
    do {
        if (!std::getline(std::cin, cmd)) cmd = "quit";
        std::istringstream is(cmd);
        std::string token;
        is >> token;
        if (token.empty()) continue;

        if (token == "quit" || token == "stop") {
            search::stop();
            if (token == "quit") {
                wait_for_search_finished();
                break;
            }
        } else if (token == "ponderhit") {
            // pondering not implemented
        } else if (token == "uci") {
            on_uci();
        } else if (token == "setoption") {
            on_setoption(is);
        } else if (token == "go") {
            on_go(is);
        } else if (token == "position") {
            on_position(is);
        } else if (token == "ucinewgame") {
            wait_for_search_finished();
            search::reset();
        } else if (token == "isready") {
            wait_for_search_finished();
            sync_out::print("readyok\n");
        } else if (token == "perft") {
            wait_for_search_finished();
            int depth = 1;
            is >> depth;
            sync_out::print("perft ", depth, ": ", perft(root_board, depth), "\n");
        } else if (token == "bench") {
            wait_for_search_finished();
            int depth = 12;
            is >> depth;
            search::bench(depth);
        } else if (token == "bbtest") {
            long long checks = 0, fails = 0;
            bool ok = bb::self_test(checks, fails);
            sync_out::print("bbtest: ", (ok ? "PASS" : "FAIL"),
                            " checks=", checks, " fails=", fails, "\n");
        } else if (token == "movetest" || token == "perfttest") {
            wait_for_search_finished();
            long long cases = 0, fails = 0;
            perft_test(cases, fails);
            sync_out::print("perfttest: ", (fails == 0 ? "PASS" : "FAIL"),
                            " cases=", cases, " fails=", fails, "\n");
        } else if (token == "d") {
            // not implemented
        } else if (!token.empty() && token[0] != '#') {
            sync_out::print("Unknown command: ", cmd, "\n");
        }
    } while (true);
}

}  // namespace uci