#pragma once

#include "types.h"
#include "board.h"

#include <cstdint>

enum GenType { ALL = 0, CAPTURES, QUIETS };

template <GenType T>
struct MoveList {
    Move moves[256];
    int32_t scores[256];
    int size = 0;

    void add(Move m) { moves[size++] = m; }
    Move* begin() { return moves; }
    Move* end() { return moves + size; }
};

using MoveListAll = MoveList<ALL>;
using MoveListCap = MoveList<CAPTURES>;
using MoveListQuiet = MoveList<QUIETS>;

template <GenType T>
void generate(const Board& b, MoveList<T>& list);

void generate_legal(const Board& b, MoveListAll& list);