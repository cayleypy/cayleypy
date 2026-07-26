"""Regression baseline benchmarks for beam search.

These are **regression baselines**, not performance targets. They exist so that a
future performance optimization plan can compare before/after and prove no regression.

Usage::

    # Run benchmarks and save baseline to .benchmarks/ (gitignored):
    pytest --benchmark-only

    # Compare against saved baseline after making perf changes:
    pytest --benchmark-only --benchmark-compare

The benchmarks use small, fast, deterministic configurations (fixed seed via conftest,
small beam_width, bounded max_steps) so each runs in under ~2 seconds on CPU.
"""

import numpy as np
import torch

from ..cayley_graph import CayleyGraph
from ..graphs_lib import PermutationGroups, prepare_graph
from ..puzzles import Puzzles

# Graph singletons constructed once at import. This file is only collected when
# explicitly passed to pytest, so import-time construction is acceptable. pytest-benchmark
# times only the benchmark() call body, so construction here does not affect measurements.
# beam_search reads graph attributes without mutating them, so sharing across benchmarks
# is safe. (Module-level singletons avoid the pytest-fixture `redefined-outer-name`
# warning that fixture-parameter shadowing would trigger.)
_LRX8_GRAPH = CayleyGraph(PermutationGroups.lrx(8))
_LRX16_GRAPH = CayleyGraph(PermutationGroups.lrx(16))
_LRX32_GRAPH = CayleyGraph(PermutationGroups.lrx(32))
_CUBE222_GRAPH = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"))
_CUBE333_GRAPH = CayleyGraph(
    Puzzles.rubik_cube(3, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)
_CUBE444_GRAPH = CayleyGraph(
    Puzzles.rubik_cube(4, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)
_CUBE555_GRAPH = CayleyGraph(
    Puzzles.rubik_cube(5, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)

# A fixed far-from-central state for LRX(8) — deterministic, no random scrambling.
_LRX8_START = [3, 5, 7, 1, 0, 6, 4, 2]

# A fixed far-from-central state for LRX(16).
_LRX16_START = list(range(10, 16)) + list(range(0, 10))

# LRX(32): last 16 elements rotated to the front (16 L-moves from central).
# keep in sync with kaggle_benchmarks/perf/run.py _LRX32_START.
_LRX32_START = list(range(16, 32)) + list(range(0, 16))

# Hardcoded scramble from ivankolt/hamming-beamsearch (54 stickers, 6 colors 0-5).
# keep in sync with kaggle_benchmarks/baseline/run.py _CUBE333_START.
# fmt: off
_CUBE333_START = [3, 3, 1, 0, 0, 2, 1, 0, 4, 4, 2, 0, 5, 1, 4, 5, 5, 3,
                  3, 3, 5, 0, 2, 5, 4, 2, 0, 2, 4, 0, 2, 3, 3, 2, 5, 5,
                  2, 1, 0, 0, 4, 1, 2, 4, 1, 4, 3, 5, 1, 5, 1, 3, 4, 1]
# fmt: on

# Deterministic scrambles for cube444 (96 stickers, 12 moves) and cube555 (150
# stickers, 10 moves), generated once via random_walks(seed=12345) and hardcoded
# so neither the benchmark nor the Kaggle kernel depends on import-time RNG state.
# Short scrambles so the search finds the solution within max_steps=30 using only
# the hamming predictor (no NN model loaded in benchmarks).
# keep in sync with kaggle_benchmarks/perf/run.py _CUBE444_START / _CUBE555_START.
# fmt: off
_CUBE444_START = [1, 5, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 4, 0, 0, 0, 5, 1,
                  4, 4, 1, 4, 1, 1, 2, 1, 1, 2, 4, 1, 4, 4, 1, 5, 5, 2,
                  2, 5, 5, 2, 3, 3, 2, 0, 1, 1, 1, 1, 3, 3, 4, 2, 3, 3,
                  2, 2, 4, 3, 4, 4, 2, 5, 3, 2, 0, 0, 3, 3, 0, 0, 3, 4,
                  1, 0, 1, 1, 3, 3, 3, 0, 3, 0, 3, 5, 5, 5, 5, 5, 5, 4,
                  4, 5, 5, 4, 4, 5]
# fmt: on
# fmt: off
_CUBE555_START = [5, 5, 5, 5, 5, 2, 2, 2, 0, 2, 3, 0, 0, 0, 0, 2, 2, 2,
                  2, 2, 3, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                  1, 1, 1, 1, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1, 2, 5, 2, 5,
                  4, 2, 5, 2, 5, 4, 2, 5, 2, 5, 4, 2, 4, 4, 5, 4, 2, 1,
                  2, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  0, 1, 1, 1, 1, 5, 5, 5, 5, 4, 2, 0, 0, 0, 0, 2, 0, 4,
                  0, 4, 2, 0, 4, 0, 4, 2, 0, 2, 2, 4, 3, 3, 4, 3, 4, 1,
                  5, 5, 5, 5, 0, 4, 4, 4, 4, 1, 5, 5, 5, 5, 0, 4, 4, 5,
                  4, 0, 4, 4, 4, 4]
# fmt: on


def bench_simple_lrx8(benchmark):
    """Benchmark simple beam search on LRX(8)."""
    result = benchmark(
        _LRX8_GRAPH.beam_search,
        start_state=_LRX8_START,
        beam_mode="simple",
        beam_width=10**5,
        max_steps=30,
    )
    assert result.path_found


def bench_advanced_lrx8_history2(benchmark):
    """Benchmark advanced beam search with history_depth=2 on LRX(8)."""
    result = benchmark(
        _LRX8_GRAPH.beam_search,
        start_state=_LRX8_START,
        beam_mode="advanced",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )
    assert result.path_found


def bench_iterated_lrx8_history2(benchmark):
    """Benchmark iterated beam search with history_depth=2 on LRX(8)."""
    result = benchmark(
        _LRX8_GRAPH.beam_search,
        start_state=_LRX8_START,
        beam_mode="iterated",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )
    assert result.path_found


def bench_advanced_lrx16_hamming(benchmark):
    """Benchmark advanced beam search with hamming predictor on LRX(16)."""
    result = benchmark(
        _LRX16_GRAPH.beam_search,
        start_state=_LRX16_START,
        beam_mode="advanced",
        beam_width=10**5,
        max_steps=20,
        history_depth=2,
    )
    assert result.path_found


def bench_simple_cube222(benchmark):
    """Benchmark simple beam search on 2x2x2 cube."""
    np.random.seed(12345)
    start_state = _CUBE222_GRAPH.random_walks(width=1, length=20)[0][-1]
    result = benchmark(
        _CUBE222_GRAPH.beam_search,
        start_state=start_state,
        beam_mode="simple",
        beam_width=10**5,
        max_steps=20,
    )
    assert result.path_found


def bench_simple_cube333(benchmark):
    """Benchmark simple beam search on 3x3x3 cube (quick level).

    path_found is not guaranteed at quick level (reduced params, no MITM); this
    benchmark measures search throughput, not solution quality. The Kaggle script
    (kaggle_benchmarks/baseline/run.py) records path_found/path_length as stats.
    """
    benchmark(
        _CUBE333_GRAPH.beam_search,
        start_state=_CUBE333_START,
        beam_mode="simple",
        beam_width=10**5,
        max_steps=30,
    )


def bench_advanced_cube333_history2(benchmark):
    """Benchmark advanced beam search with history_depth=2 on 3x3x3 cube (quick level)."""
    benchmark(
        _CUBE333_GRAPH.beam_search,
        start_state=_CUBE333_START,
        beam_mode="advanced",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )


def bench_iterated_cube333_history2(benchmark):
    """Benchmark iterated beam search with history_depth=2 on 3x3x3 cube (quick level)."""
    benchmark(
        _CUBE333_GRAPH.beam_search,
        start_state=_CUBE333_START,
        beam_mode="iterated",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )


def bench_iterated_lrx32_history2(benchmark):
    """Benchmark iterated beam search on LRX(32) (Phase 0.3 profiling group).

    LRX(32) has only 3 generators, so per-chunk overhead is isolated from the
    n_generators-scaled work that dominates cube groups. The 16-move scramble
    (half-rotation) solves in 16 steps at bw=10^5 with the hamming predictor.
    """
    result = benchmark(
        _LRX32_GRAPH.beam_search,
        start_state=_LRX32_START,
        beam_mode="iterated",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )
    assert result.path_found


def bench_iterated_cube444_history2(benchmark):
    """Benchmark iterated beam search on 4x4x4 cube (Phase 0.3 profiling group).

    cube444 has 24 generators (2x cube333), so per-chunk overhead scales 2x.
    12-move scramble solves within max_steps=30 with the hamming predictor at
    GPU beams (bw=2^18); at the CPU benchmark beam (bw=10^4) path_found is not
    guaranteed (throughput benchmark, matching the cube333_quick pattern).
    Uses bw=10^4 (not 10^5) because cube444's 2x generators + 2x state_size make
    bw=10^5 ~40 min on CPU; bw=10^4 keeps it under 30s while still detecting
    regressions (beam_width only needs to be consistent across runs, not match
    the Kaggle config).
    """
    benchmark(
        _CUBE444_GRAPH.beam_search,
        start_state=_CUBE444_START,
        beam_mode="iterated",
        beam_width=10**4,
        max_steps=30,
        history_depth=2,
    )


def bench_iterated_cube555_history2(benchmark):
    """Benchmark iterated beam search on 5x5x5 cube (Phase 0.3 profiling group).

    cube555 has the largest state_size (150), so hashing/sort/memory dominate.
    10-move scramble solves within max_steps=30 with the hamming predictor at
    GPU beams (bw=2^18); at the CPU benchmark beam (bw=10^4) path_found is not
    guaranteed (throughput benchmark, matching the cube333_quick pattern).
    Uses bw=10^4 (not 10^5) because cube555's 24 generators + 150 state_size make
    bw=10^5 ~60 min on CPU; bw=10^4 keeps it under 60s while still detecting
    regressions.
    """
    benchmark(
        _CUBE555_GRAPH.beam_search,
        start_state=_CUBE555_START,
        beam_mode="iterated",
        beam_width=10**4,
        max_steps=30,
        history_depth=2,
    )
