import pytest

from cayleypy import load_dataset, CayleyGraph, PermutationGroups
from cayleypy.algo import bfs_bitmask


@pytest.mark.unit
def test_bfs_bitmask_lx9():
    graph = CayleyGraph(PermutationGroups.lx(9))
    result = bfs_bitmask(graph)
    assert result == load_dataset("lx_cayley_growth")["9"]


@pytest.mark.unit
def test_bfs_bitmask_lrx_10_first_5_layers():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    result = bfs_bitmask(graph, max_diameter=5)
    assert result == load_dataset("lrx_cayley_growth")["10"][:6]


@pytest.mark.unit
@pytest.mark.slow
def test_bfs_bitmask_lrx_10():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    assert bfs_bitmask(graph) == load_dataset("lrx_cayley_growth")["10"]


@pytest.mark.unit
@pytest.mark.slow
def test_bfs_bitmask_pancake_9():
    graph = CayleyGraph(PermutationGroups.pancake(9))
    assert bfs_bitmask(graph) == load_dataset("pancake_cayley_growth")["9"]
