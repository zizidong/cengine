#pragma once

#include "types.h"

namespace zobrist {

void init();

extern Key psq[PIECE_NB][SQ_NB];
extern Key castling[16];
extern Key ep_file[8];
extern Key side;

}  // namespace zobrist