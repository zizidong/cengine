#pragma once

#include "nnue_full.h"

#include "types.h"
#include "board.h"

#include <string>

namespace nnue {

inline void init() {}

inline bool load(const std::string& path) { return nnue_full::load(path); }
inline bool is_loaded() { return nnue_full::is_loaded(); }
inline const std::string& loaded_path() { return nnue_full::loaded_path(); }

inline bool load_small(const std::string& path) { return nnue_full::load_small(path); }
inline bool small_loaded() { return nnue_full::small_loaded(); }
inline const std::string& small_path() { return nnue_full::small_path(); }
inline bool big_net_loaded() { return nnue_full::big_net_loaded(); }
inline void set_big_net_ply(int v) { nnue_full::set_big_net_ply(v); }
inline void set_big_net_pv(int v) { nnue_full::set_big_net_pv(v); }
inline int  big_net_pv() { return nnue_full::big_net_pv(); }
inline int  big_net_ply() { return nnue_full::big_net_ply(); }

inline int evaluate(const Board& b) { return nnue_full::evaluate(b); }

inline void acc_set_root(const Board& b) { nnue_full::acc_set_root(b); }
inline void acc_push(const Board& b, bool big) { nnue_full::acc_push(b, big); }
inline void acc_pop() { nnue_full::acc_pop(); }

inline nnue_full::AccStats acc_stats() { return nnue_full::acc_stats(); }
inline void acc_cycles(uint64_t& b, uint64_t& a, uint64_t& f) { nnue_full::acc_cycles(b, a, f); }
inline void acc_build_cycles(uint64_t& ap, uint64_t& s) { nnue_full::acc_build_cycles(ap, s); }
inline uint64_t tsc_now() { return nnue_full::tsc_now(); }

// Backward-compatible wrappers (unused but harmless).
inline std::string auto_load_from_dir(const std::string&) { return {}; }
inline bool auto_load_near_executable() { return false; }
inline void update_acc(const Board&) {}

}  // namespace nnue