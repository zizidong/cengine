#pragma once

#include "board.h"

#include <cstdint>

uint64_t perft(const Board& b, int depth);

// Runs the standard perft suite and compares against known-good node counts.
// Returns the number of failing cases; fills `cases` and `fails`.
int perft_test(long long& cases, long long& fails);