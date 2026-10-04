cmake_minimum_required(VERSION 3.12)

project(libchess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif ()

add_library(libchess INTERFACE)
target_include_directories(libchess INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_compile_features(libchess INTERFACE cxx_std_17)

add_library(libchess::libchess ALIAS libchess)