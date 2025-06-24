import os

import pytest

from cayleypy import load_dataset, bfs_bitmask, CayleyGraph, PermutationGroups

FAST_RUN = os.getenv("FAST") == "1"


def test_bfs_bitmask_lrx_10_first_5_layers():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    result = bfs_bitmask(graph, max_diameter=5)
    assert result == load_dataset("lrx_cayley_growth")["10"][:6]


@pytest.mark.skipif(FAST_RUN, reason="slow test")
def test_bfs_bitmask_lrx_10():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    assert bfs_bitmask(graph) == load_dataset("lrx_cayley_growth")["10"]


@pytest.mark.skipif(FAST_RUN, reason="slow test")
def test_bfs_bitmask_pancake_9():
    graph = CayleyGraph(PermutationGroups.pancake(9))
    assert bfs_bitmask(graph) == load_dataset("pancake_cayley_growth")["9"]
