#pragma once

#include "types.h"
#include "board.h"

namespace eval {

void init();
int evaluate(const Board& b);

// Incremental NNUE accumulator hooks (no-ops when NNUE is not loaded).
void nnue_set_root(const Board& b);
void nnue_on_make(const Board& b, bool big);
void nnue_on_unmake();

// optional NNUE integration
bool try_load_nnue(const char* path);
bool try_load_nnue_small(const char* path);
bool using_nnue();
bool using_nnue_small();
void nnue_set_big_net_ply(int v);
int  nnue_big_net_ply();
void nnue_set_big_net_pv(int v);
int  nnue_big_net_pv();

constexpr int PIECE_VALUE[PIECE_TYPE_NB] = { 100, 320, 330, 500, 900, 0 };

}  // namespace eval