# cengine

cengine is a UCI chess engine written in C++23.

## Features

- Board representation: bitboards with precomputed attack tables. Sliding
  attacks use a PEXT/BMI2 lookup table when built with `CENGINE_PEXT`, and a
  ray-based fallback otherwise.
- Search: iterative deepening alpha-beta with principal variation search,
  aspiration windows, and a clustered transposition table. Pruning and
  reduction: null-move pruning, reverse futility pruning, futility pruning,
  late move pruning, SEE pruning of captures, late move reductions, check
  extensions with a per-line budget, and internal iterative reduction. Move
  ordering uses the hash move, MVV-LVA captures, killers, a counter move, and
  history.
- Quiescence search generates check evasions when in check and prunes losing
  captures with SEE.
- Draw handling: fifty-move rule, repetition (a position repeated within the
  search is treated as a draw), and dead-position material (insufficient
  material).
- Evaluation: NNUE. A small HalfKAv2_hm network (128 wide, piece-square only)
  is used for most nodes, and a large HalfKAv2_hm + FullThreats + PP_3Wide
  network (L1 1024, L2 32, L3 32) is used at PV nodes and shallow plies. The
  feature accumulators are updated incrementally. If no network is available,
  evaluation falls back to material plus piece-square tables.
- Threads: lazy SMP.
- UCI interface with `bench`, `perft`, `perfttest`, and `bbtest` commands.

## Building

Requirements:

- CMake 3.20 or newer
- A C++23 compiler (GCC 13+, Clang 16+, or MSVC 19.30+)
- zlib
- Network access on the first configure (CMake FetchContent downloads libchess)

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

The executable is written to `build/bin/cengine`.

CMake options:

- `CENGINE_NATIVE` (default ON): build with `-march=native`. Turn this off to
  produce a binary that can run on other machines.
- `CENGINE_PEXT` (default OFF): enable the BMI2/PEXT sliding attack tables.

## Neural networks

The engine loads two network files:

- `nn-37f18f62d772.nnue`: small network, included in `src/nnue/`.
- `nn-134a887f4c8f.nnue`: large network, not included in this repository
  because of its size (about 94 MB).

Search path for a network file: the executable's directory, then `src/nnue/`
relative to the executable directory and to the current working directory. To
enable NNUE evaluation, place the large network next to the small one in
`src/nnue/`, or in the same directory as the executable. Without the large
network, the engine evaluates with material and piece-square tables.

## Usage

Run the executable and type UCI commands on standard input.

UCI options:

- `Hash` (spin, default 32, min 1, max 1024): transposition table size in MB.
- `Threads` (spin, default 1, min 1, max 64): number of search threads.
- `Move Overhead` (spin, default 10): accepted for compatibility.
- `BigNetPly` (spin, default 2, min 0, max 64): use the large network up to
  this ply.
- `BigNetPV` (spin, default 1, min 0, max 1): use the large network at PV
  nodes.
- `EvalFile` (string): path to a network file.

Additional commands:

- `bench [depth]`: run the built-in benchmark (default depth 12).
- `perft <depth>`: run perft from the current position.
- `perfttest`: run the built-in perft test suite.
- `bbtest`: run the attack and bitboard self test.

## Notes

- `searchmoves` in `go` is parsed but not used, and pondering is not
  implemented.
