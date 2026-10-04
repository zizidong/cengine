#pragma once

#include "types.h"
#include "board.h"

#include <cstdint>

enum TTFlag : uint8_t {
    TT_NONE  = 0,
    TT_EXACT = 1,
    TT_LOWER = 2,
    TT_UPPER = 3
};

struct TTEntry {
    uint32_t key32;     // upper 32 bits of the position key, for verification
    uint16_t move16;    // packed move (see Move); 0 means "no move"
    int16_t  score;
    int16_t  eval;
    int8_t   depth;
    uint8_t  genBound;  // generation (high bits) | bound (low 2 bits)
    uint16_t pad;
};

static_assert(sizeof(TTEntry) == 16, "TTEntry should be 16 bytes");

constexpr int TT_CLUSTER = 4;

class TranspositionTable {
public:
    TranspositionTable() = default;
    ~TranspositionTable() { clear(); }

    void resize(uint64_t mb);
    void clear();
    void clear_entries();  // clear contents but keep the allocation

    void new_search() { generation = uint8_t(generation >= 248 ? 8 : generation + 8); }

    // Returns the entry with the best matching depth (or null if no hit).
    TTEntry* probe(Key key, bool& found) const;
    void store(Key key, Move move, int score, int eval, int depth, int flag);

    uint64_t size_mb() const { return count * sizeof(TTEntry) / (1024 * 1024); }

private:
    TTEntry* entries = nullptr;
    uint64_t count = 0;       // number of clusters
    uint8_t  generation = 0;
};

extern TranspositionTable TT;

inline uint8_t tt_bound(const TTEntry* e) { return uint8_t(e->genBound & 0x03); }
inline uint8_t tt_gen(const TTEntry* e)   { return uint8_t(e->genBound & 0xF8); }

// Signature over the payload fields.  The stored key32 is xored with this, so a
// torn/lockless-racing read (mismatched key + data) fails the key check and is
// rejected instead of returning a wrong score/move.
inline uint32_t tt_signature(const TTEntry* e) {
    const uint32_t a = uint32_t(e->move16) | (uint32_t(uint16_t(e->score)) << 16);
    const uint32_t b = uint32_t(uint16_t(e->eval)) | (uint32_t(uint8_t(e->depth)) << 16)
                     | (uint32_t(e->genBound) << 24);
    return (a * 0x9E3779B1u) ^ (b * 0x85EBCA77u);
}

inline bool tt_matches(const TTEntry* e, uint32_t k32) {
    return e->genBound != 0 && e->key32 == (k32 ^ tt_signature(e));
}