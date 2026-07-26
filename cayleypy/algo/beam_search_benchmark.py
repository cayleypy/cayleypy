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
_CUBE222_GRAPH = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"))
_CUBE333_GRAPH = CayleyGraph(
    Puzzles.rubik_cube(3, metric="QTM"),
    dtype=torch.int8,
    bit_encoding_width=None,
    hash_chunk_size=2**16,
)

# A fixed far-from-central state for LRX(8) — deterministic, no random scrambling.
_LRX8_START = [3, 5, 7, 1, 0, 6, 4, 2]

# A fixed far-from-central state for LRX(16).
_LRX16_START = list(range(10, 16)) + list(range(0, 10))

# Hardcoded scramble from ivankolt/hamming-beamsearch (54 stickers, 6 colors 0-5).
# keep in sync with kaggle_benchmarks/baseline/run.py _CUBE333_START.
# fmt: off
_CUBE333_START = [3, 3, 1, 0, 0, 2, 1, 0, 4, 4, 2, 0, 5, 1, 4, 5, 5, 3,
                  3, 3, 5, 0, 2, 5, 4, 2, 0, 2, 4, 0, 2, 3, 3, 2, 5, 5,
                  2, 1, 0, 0, 4, 1, 2, 4, 1, 4, 3, 5, 1, 5, 1, 3, 4, 1]
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
