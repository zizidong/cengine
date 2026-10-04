#include "nnue.h"

#include "evaluate.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace nnue {

namespace {

constexpr int OUTPUT_SCALE = 400;
constexpr int WEIGHT_SCALE = 64;
constexpr int I16_SCALE    = 16;

int16_t clamp_i16(int x) { return int16_t(std::clamp(x, 0, 0x7FFF)); }

struct Weights {
    int input_size;
    int hidden_size;
    int output_buckets;

    int ft_weights_quant;
    int ft_biases_quant;
    int out_weights_quant;
    int out_biases_quant;

    std::vector<int16_t> ft_weights;
    std::vector<int16_t> ft_biases;
    std::vector<int32_t> out_weights;
    std::vector<int32_t> out_biases;

    std::vector<int16_t> acc[2];
};

Weights g;

bool g_loaded = false;
std::string g_loaded_path;

uint32_t read_u32_be(FILE* f) {
    unsigned char b[4];
    if (std::fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

uint32_t read_u32_le(FILE* f) {
    unsigned char b[4];
    if (std::fread(b, 1, 4, f) != 4) return 0;
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

int16_t read_i16_le(FILE* f) {
    unsigned char b[2];
    if (std::fread(b, 1, 2, f) != 2) return 0;
    return int16_t(b[0] | (b[1] << 8));
}

int8_t read_i8(FILE* f) {
    return int8_t(std::fgetc(f));
}

struct ArchInfo {
    int input_size;
    int hidden_size;
    int buckets;
};

bool lookup_arch(uint32_t hash, ArchInfo& out) {
    switch (hash) {
        case 0x7AF32F16u:
            out = {10 * 64 * 64, 256, 8};
            return true;
        case 0xFA8A446Au:
            out = {11 * 64 * 64, 512, 8};
            return true;
    }
    return false;
}

bool load_halfkp(FILE* f, int input_size, int hidden_size, int buckets) {
    g.input_size = input_size;
    g.hidden_size = hidden_size;
    g.output_buckets = buckets;

    g.ft_weights.resize(size_t(input_size) * hidden_size);
    g.ft_biases.resize(hidden_size);
    g.out_biases.resize(buckets);
    g.out_weights.resize(size_t(2) * buckets * hidden_size);

    for (size_t i = 0; i < g.ft_weights.size(); ++i)
        g.ft_weights[i] = read_i8(f);

    for (int i = 0; i < hidden_size; ++i)
        g.ft_biases[i] = read_i16_le(f);

    for (size_t i = 0; i < g.out_weights.size(); ++i)
        g.out_weights[i] = read_i8(f);

    for (int i = 0; i < buckets; ++i)
        g.out_biases[i] = int32_t(read_u32_le(f));

    g.acc[0].assign(hidden_size, 0);
    g.acc[1].assign(hidden_size, 0);
    return true;
}

}  // namespace

void init() {
    g.acc[0].clear();
    g.acc[1].clear();
    g_loaded = false;
    g_loaded_path.clear();
}

bool load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t version   = read_u32_be(f);
    uint32_t hash      = read_u32_be(f);
    uint32_t desc_len  = read_u32_le(f);
    (void)hash;

    if (desc_len) {
        std::vector<char> desc(desc_len);
        if (std::fread(desc.data(), 1, desc_len, f) != desc_len) {
            std::fclose(f);
            return false;
        }
    }

    ArchInfo arch{};
    if (!lookup_arch(version, arch)) {
        std::fclose(f);
        return false;
    }

    bool ok = load_halfkp(f, arch.input_size, arch.hidden_size, arch.buckets);
    std::fclose(f);
    g_loaded = ok;
    if (ok) g_loaded_path = path;
    return ok;
}

bool is_loaded() { return g_loaded; }

const std::string& loaded_path() { return g_loaded_path; }

std::string auto_load_from_dir(const std::string& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec)) return {};

    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!e.is_regular_file(ec)) continue;
        auto p = e.path();
        if (p.extension() == ".nnue") files.push_back(p);
    }
    if (files.empty()) return {};

    std::sort(files.begin(), files.end());

    for (auto& p : files) {
        if (load(p.string())) {
            return p.string();
        }
    }
    return {};
}

bool auto_load_near_executable() {
    namespace fs = std::filesystem;
    std::error_code ec;

    auto try_dir = [&](const fs::path& p) -> bool {
        if (fs::is_directory(p, ec)) {
            std::string got = auto_load_from_dir(p.string());
            if (!got.empty()) return true;
        }
        return false;
    };

    fs::path here = fs::current_path(ec);
    if (try_dir(here / "nnue")) return true;
    if (try_dir(here / "src" / "nnue")) return true;
    if (try_dir("../src/nnue")) return true;
    if (try_dir("../../src/nnue")) return true;
    return false;
}

namespace {
inline int feature_index(int king_persp, int piece_persp, Piece p) {
    int pc = (color_of(p) == WHITE) ? type_of(p) : type_of(p) + 6;
    return king_persp * 64 * 10 + piece_persp * 10 + pc;
}
}  // namespace

static void refresh_accumulator(const Board& b, int perspective) {
    Color us = Color(perspective);
    Square ksq = b.king_square(us);
    int ki = us == WHITE ? ksq : (ksq ^ 56);

    auto& acc = g.acc[perspective];
    for (int i = 0; i < g.hidden_size; ++i)
        acc[i] = g.ft_biases[i];

    int idxs[32];
    int cnt = 0;
    for (int c = 0; c < COLOR_NB; ++c) {
        Bitboard bb = b.pieces(Color(c));
        while (uint64_t(bb) != 0) {
            Square s = Square(__builtin_ctzll(uint64_t(bb)));
            bb = bb & (bb - Bitboard(1));
            Piece p = b.piece_on(s);
            int ps = us == WHITE ? s : (s ^ 56);
            idxs[cnt++] = feature_index(ki, ps, p);
        }
    }

    for (int i = 0; i < g.hidden_size; ++i) {
        int32_t sum = acc[i];
        for (int j = 0; j < cnt; ++j) {
            int f = idxs[j];
            sum += g.ft_weights[size_t(f) * g.hidden_size + i];
        }
        acc[i] = clamp_i16(sum);
    }
}

void update_acc(const Board& b) {
    if (!g_loaded) return;
    refresh_accumulator(b, b.side_to_move());
}

int evaluate(const Board& b) {
    if (!g_loaded)
        return eval::PIECE_VALUE[0];

    Color us = b.side_to_move();
    refresh_accumulator(b, us);

    int bucket = 0;
    if (g.output_buckets > 1) {
        int cnt = __builtin_popcountll(uint64_t(b.occupancy()));
        bucket = std::clamp((cnt - 2) / 4, 0, g.output_buckets - 1);
    }

    const int16_t* acc = g.acc[us].data();
    const int32_t* ow = g.out_weights.data() + size_t(us) * g.output_buckets * g.hidden_size + size_t(bucket) * g.hidden_size;

    int64_t sum = g.out_biases[bucket];
    for (int i = 0; i < g.hidden_size; ++i)
        sum += int64_t(acc[i]) * ow[i];

    int score = int((sum * OUTPUT_SCALE) / (WEIGHT_SCALE * I16_SCALE));
    if (score >  4000) score =  4000;
    if (score < -4000) score = -4000;
    return us == WHITE ? score : -score;
}

}  // namespace nnue
