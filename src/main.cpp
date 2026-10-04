#include "uci.h"

#include "bitboard.h"
#include "nnue.h"
#include "search.h"
#include "tt.h"
#include "zobrist.h"

#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string exe_dir(const char* argv0) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path p = fs::absolute(argv0 ? argv0 : "", ec);
    if (ec || p.empty()) return {};
    fs::path dir = p.parent_path();
    if (dir.empty()) dir = fs::current_path(ec);
    return dir.string();
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    bb::init();
    zobrist::init();
    nnue::init();
    search::init();
    TT.resize(32);

    const char* name = "nn-134a887f4c8f.nnue";
    const char* sname = "nn-37f18f62d772.nnue";
    std::string dir = exe_dir(argv[0]);

    std::vector<std::string> candidates;
    if (!dir.empty()) {
        candidates.push_back(dir + "/" + name);
        candidates.push_back(dir + "/../src/nnue/" + name);
        candidates.push_back(dir + "/../../src/nnue/" + name);
    }
    candidates.push_back(name);
    candidates.push_back(std::string("src/nnue/") + name);
    candidates.push_back(std::string("../src/nnue/") + name);
    candidates.push_back(std::string("../../src/nnue/") + name);

    for (const std::string& c : candidates)
        if (nnue::load(c))
            break;

    // Small (cache-resident) net for the small/big lazy scheme.
    std::vector<std::string> scandidates;
    if (!dir.empty()) {
        scandidates.push_back(dir + "/" + sname);
        scandidates.push_back(dir + "/../src/nnue/" + sname);
        scandidates.push_back(dir + "/../../src/nnue/" + sname);
    }
    scandidates.push_back(sname);
    scandidates.push_back(std::string("src/nnue/") + sname);
    scandidates.push_back(std::string("../src/nnue/") + sname);
    scandidates.push_back(std::string("../../src/nnue/") + sname);

    for (const std::string& c : scandidates)
        if (nnue::load_small(c))
            break;

    uci::loop();
    return 0;
}