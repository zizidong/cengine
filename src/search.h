#pragma once

#include "types.h"
#include "board.h"
#include "movegen.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

struct SearchLimits {
    int depth = -1;
    int movetime = -1;
    int wtime = -1;
    int btime = -1;
    int winc = 0;
    int binc = 0;
    int movestogo = 0;
    int nodes = -1;
    uint64_t move_mask = 0;  // "searchmoves"
    bool infinite = false;

    bool has_time() const { return movetime >= 0 || wtime >= 0 || btime >= 0; }
};

struct SearchResult {
    Move best;
    Move ponder;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
};

namespace search {

void init();
void reset();

void set_position(const Board& b);
void go(SearchLimits limits, SearchResult& result);
void stop();
void clear_stop();

// Number of search threads (Lazy SMP). Default 1.
void set_threads(int n);

// Runs a fixed suite of positions at the given depth and prints total nodes/nps.
void bench(int depth);

extern bool stopped;

}  // namespace search