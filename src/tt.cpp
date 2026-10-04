#include "tt.h"

#include <algorithm>

#include <cstdlib>

TranspositionTable TT;

void TranspositionTable::resize(uint64_t mb) {
    clear();
    if (mb < 1) mb = 1;
    uint64_t bytes = mb * 1024 * 1024;
    uint64_t entry_count = bytes / sizeof(TTEntry);
    count = entry_count / TT_CLUSTER;
    if (count == 0) count = 1;
    entries = static_cast<TTEntry*>(std::malloc(count * TT_CLUSTER * sizeof(TTEntry)));
    if (!entries) {
        count = 0;
        return;
    }
    for (uint64_t i = 0; i < count * TT_CLUSTER; ++i)
        entries[i] = TTEntry{};
}

void TranspositionTable::clear() {
    if (entries) {
        std::free(entries);
        entries = nullptr;
    }
    count = 0;
    generation = 0;
}

void TranspositionTable::clear_entries() {
    if (!entries) return;
    const uint64_t total = count * TT_CLUSTER;
    for (uint64_t i = 0; i < total; ++i)
        entries[i] = TTEntry{};
    generation = 0;
}

TTEntry* TranspositionTable::probe(Key key, bool& found) const {
    found = false;
    if (!entries || count == 0) return nullptr;

    TTEntry* cluster = &entries[(key % count) * TT_CLUSTER];
    const uint32_t k32 = uint32_t(key >> 32);

    TTEntry* replace = &cluster[0];
    for (int i = 0; i < TT_CLUSTER; ++i) {
        TTEntry* e = &cluster[i];
        if (tt_matches(e, k32)) {
            found = true;
            return e;
        }
        // choose least valuable entry to replace
        if (e->genBound == 0) { replace = e; break; }
        if (e->depth < replace->depth) replace = e;
    }
    return replace;
}

void TranspositionTable::store(Key key, Move move, int score, int eval, int depth, int flag) {
    if (!entries || count == 0) return;

    TTEntry* cluster = &entries[(key % count) * TT_CLUSTER];
    const uint32_t k32 = uint32_t(key >> 32);
    const uint8_t gb = uint8_t(generation | uint8_t(flag));

    TTEntry* slot = nullptr;
    TTEntry* replace = &cluster[0];
    for (int i = 0; i < TT_CLUSTER; ++i) {
        TTEntry* e = &cluster[i];
        if (tt_matches(e, k32)) { slot = e; break; }
        if (e->genBound == 0) { if (!slot) slot = e; }
        if (e->depth < replace->depth) replace = e;
    }
    if (!slot) {
        // Prefer replacing an old-generation entry of lesser depth.
        slot = replace;
        for (int i = 0; i < TT_CLUSTER; ++i) {
            TTEntry* e = &cluster[i];
            if (tt_gen(e) != generation && e->depth <= depth) { slot = e; break; }
        }
    }

    // Keep an existing move if the new search has none.
    if (move == Move::none() && tt_matches(slot, k32))
        move = Move(slot->move16);

    slot->move16  = uint16_t(move.raw());
    slot->score   = int16_t(score);
    slot->eval    = int16_t(eval);
    slot->depth   = int8_t(std::clamp(depth, -127, 127));
    slot->genBound = gb;
    slot->key32   = k32 ^ tt_signature(slot);
}