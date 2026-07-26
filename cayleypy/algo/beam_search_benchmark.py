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
import pytest

from ..cayley_graph import CayleyGraph
from ..graphs_lib import PermutationGroups, prepare_graph


@pytest.fixture
def lrx8():
    return CayleyGraph(PermutationGroups.lrx(8))


@pytest.fixture
def lrx16():
    return CayleyGraph(PermutationGroups.lrx(16))


@pytest.fixture
def cube222():
    return CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"))


# A fixed far-from-central state for LRX(8) — deterministic, no random scrambling.
_LRX8_START = [3, 5, 7, 1, 0, 6, 4, 2]

# A fixed far-from-central state for LRX(16).
_LRX16_START = list(range(10, 16)) + list(range(0, 10))


def bench_simple_lrx8(benchmark, lrx8):
    """Benchmark simple beam search on LRX(8)."""
    result = benchmark(
        lrx8.beam_search,
        start_state=_LRX8_START,
        beam_mode="simple",
        beam_width=10**5,
        max_steps=30,
    )
    assert result.path_found


def bench_advanced_lrx8_history2(benchmark, lrx8):
    """Benchmark advanced beam search with history_depth=2 on LRX(8)."""
    result = benchmark(
        lrx8.beam_search,
        start_state=_LRX8_START,
        beam_mode="advanced",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )
    assert result.path_found


def bench_iterated_lrx8_history2(benchmark, lrx8):
    """Benchmark iterated beam search with history_depth=2 on LRX(8)."""
    result = benchmark(
        lrx8.beam_search,
        start_state=_LRX8_START,
        beam_mode="iterated",
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )
    assert result.path_found


def bench_advanced_lrx16_hamming(benchmark, lrx16):
    """Benchmark advanced beam search with hamming predictor on LRX(16)."""
    result = benchmark(
        lrx16.beam_search,
        start_state=_LRX16_START,
        beam_mode="advanced",
        beam_width=10**5,
        max_steps=20,
        history_depth=2,
    )
    assert result.path_found


def bench_simple_cube222(benchmark, cube222):
    """Benchmark simple beam search on 2x2x2 cube."""
    np.random.seed(12345)
    start_state = cube222.random_walks(width=1, length=20)[0][-1]
    result = benchmark(
        cube222.beam_search,
        start_state=start_state,
        beam_mode="simple",
        beam_width=10**5,
        max_steps=20,
    )
    assert result.path_found
