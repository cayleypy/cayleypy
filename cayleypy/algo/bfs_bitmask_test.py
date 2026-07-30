import os

import pytest

from cayleypy import load_dataset, CayleyGraph, PermutationGroups
from cayleypy.algo import bfs_bitmask

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def test_bfs_bitmask_lx9():
    graph = CayleyGraph(PermutationGroups.lx(9), bit_encoding_width="auto")
    result = bfs_bitmask(graph)
    assert result == load_dataset("lx_cayley_growth")["9"]


def test_bfs_bitmask_lrx_10_first_5_layers():
    graph = CayleyGraph(PermutationGroups.lrx(10), bit_encoding_width="auto")
    result = bfs_bitmask(graph, max_diameter=5)
    assert result == load_dataset("lrx_cayley_growth")["10"][:6]


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_bfs_bitmask_lrx_10():
    graph = CayleyGraph(PermutationGroups.lrx(10), bit_encoding_width="auto")
    assert bfs_bitmask(graph) == load_dataset("lrx_cayley_growth")["10"]


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_bfs_bitmask_pancake_9():
    graph = CayleyGraph(PermutationGroups.pancake(9), bit_encoding_width="auto")
    assert bfs_bitmask(graph) == load_dataset("pancake_cayley_growth")["9"]


def test_bfs_bitmask_n_above_16_raises():
    """A8 regression: n > 16 must raise AssertionError due to 4-bit nibble overflow."""
    graph = CayleyGraph(PermutationGroups.lrx(17), bit_encoding_width="auto")
    with pytest.raises(AssertionError, match="N<=16"):
        bfs_bitmask(graph)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_bfs_bitmask_paint_gray_single_chunk():
    """B1 regression: paint_gray with all perms in one chunk must not raise IndexError."""
    graph = CayleyGraph(PermutationGroups.lrx(9), bit_encoding_width="auto")
    # BFS with small max_diameter exercises the single-chunk paint_gray path
    # when the first layer's neighbors all share the same suffix
    result = bfs_bitmask(graph, max_diameter=3)
    assert len(result) == 4  # layers 0-3
