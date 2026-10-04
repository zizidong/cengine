cmake_minimum_required(VERSION 3.20)
project(cengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
FetchContent_Declare(
    libchess
    GIT_REPOSITORY https://github.com/Mk-Chan/libchess.git
    GIT_TAG        master
    PATCH_COMMAND  ${CMAKE_COMMAND} -E copy_if_different
                   ${CMAKE_CURRENT_SOURCE_DIR}/libchess-patch/CMakeLists.txt
                   <SOURCE_DIR>/CMakeLists.txt
)
FetchContent_MakeAvailable(libchess)

add_executable(cengine
    src/main.cpp
    src/uci.cpp
    src/evaluate.cpp
    src/nnue_full.cpp
    src/search.cpp
    src/tt.cpp
    src/perft.cpp
    src/zobrist.cpp
    src/board.cpp
    src/bitboard.cpp
    src/attack.cpp
    src/movegen.cpp
)

target_include_directories(cengine PRIVATE src)
target_link_libraries(cengine PRIVATE libchess::libchess)

# Link-time optimization so the hot cross-TU calls (Board accessors, eval
# wrappers, NNUE) can be inlined into the search.
if(NOT MSVC)
    set_property(TARGET cengine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET cengine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()

# zlib for the small (RukChess-style) net files.
find_package(ZLIB)
if(ZLIB_FOUND)
    target_link_libraries(cengine PRIVATE ZLIB::ZLIB)
else()
    target_link_libraries(cengine PRIVATE z)
endif()

option(CENGINE_PEXT "Enable x86-64 BMI2/PEXT hardware instructions" OFF)
option(CENGINE_NATIVE "Optimize for the build machine (-march=native)" ON)

if(MSVC)
    target_compile_options(cengine PRIVATE /W4 /O2)
else()
    target_compile_options(cengine PRIVATE -Wall -Wextra -Wpedantic)

    set(CENGINE_ARCH_FLAGS "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        if(CENGINE_NATIVE)
            list(APPEND CENGINE_ARCH_FLAGS -march=native)
        else()
            list(APPEND CENGINE_ARCH_FLAGS -mavx2 -mfma -mpopcnt)
        endif()
        if(CENGINE_PEXT)
            list(APPEND CENGINE_ARCH_FLAGS -mbmi2)
        endif()
    endif()

    target_compile_options(cengine PRIVATE
        $<$<CONFIG:Release>:-O3 -funroll-loops>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:Debug>:-O0 -g>
        ${CENGINE_ARCH_FLAGS}
    )
endif()
