#pragma once

#include <iostream>
#include <mutex>

namespace sync_out {

inline std::mutex& mtx() {
    static std::mutex m;
    return m;
}

template <typename... Ts>
void print(Ts&&... ts) {
    std::lock_guard<std::mutex> lock(mtx());
    (std::cout << ... << ts);
    std::cout.flush();
}

}  // namespace sync_out