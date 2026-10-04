#pragma once

// Self-contained scalar port of Stockfish's modern NNUE (HalfKAv2_hm +
// FullThreats + PP_3Wide, L1=1024 / L2=32 / L3=32) for cengine.
//
// Full-refresh evaluation from a cengine Board (no incremental accumulator).
// Uses cengine's bb:: attack tables in place of Stockfish's bitboard/attacks.

#include "types.h"
#include "board.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nnue_full {

// Load a Stockfish-style .nnue file. Returns true on success.
bool load(const std::string& path);

// Whether a network is currently loaded.
bool is_loaded();

// Evaluate from White's perspective, in centipawns. Requires a loaded network.
int evaluate(const Board& b);

// Current loaded path (empty if none).
const std::string& loaded_path();

// Incremental accumulator, driven by the searcher:
//   acc_set_root(board) before a search, acc_push(board) after each successful
//   make_move (board already updated), acc_pop() after each unmake_move.
// evaluate() uses the cached accumulator when it matches the position, and
// silently falls back to a full refresh otherwise.
void acc_set_root(const Board& b);
void acc_push(const Board& b, bool big);
void acc_pop();

// Small (cache-resident) net: RukChess-style, zlib-compressed .nnue.
bool load_small(const std::string& path);
bool small_loaded();
const std::string& small_path();
int  evaluate_small(const Board& b);

// Small/big dispatch: nodes with ply <= this (and PV nodes) use the big net.
void set_big_net_ply(int v);
int  big_net_ply();
void set_big_net_pv(int v);
int  big_net_pv();
bool big_net_loaded();

// Accumulator instrumentation: how many evaluations were served from the
// incremental accumulator vs a full refresh, plus push/diff statistics.
struct AccStats {
    uint64_t incremental_evals;
    uint64_t full_evals;
    uint64_t pushes;
    uint64_t diff_applies;
    uint64_t big_evals;
};
AccStats acc_stats();
void acc_reset_stats();

// Cycle counters for the NNUE hot path.
void acc_cycles(uint64_t& build, uint64_t& apply, uint64_t& finish);
void acc_build_cycles(uint64_t& append, uint64_t& sort);
uint64_t tsc_now();

}  // namespace nnue_full