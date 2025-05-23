from cayleypy import CayleyGraph, prepare_graph, load_dataset, bfs_numpy, bfs_bitmask
import pytest
import os

FAST_RUN = os.getenv("FAST") == "1"


def test_bfs_numpy():
    graph = prepare_graph("lrx", n=7)
    assert bfs_numpy(graph) == load_dataset("lrx_cayley_growth")["7"]

    graph = prepare_graph("top_spin", n=7)
    assert bfs_numpy(graph) == load_dataset("top_spin_cayley_growth")["7"]

    graph = prepare_graph("pancake", n=7)
    assert bfs_numpy(graph) == load_dataset("pancake_cayley_growth")["7"]

    dest = "000000000111111111"
    graph = CayleyGraph(prepare_graph("top_spin", n=18).generators, dest=dest)
    assert bfs_numpy(graph) == load_dataset("top_spin_coset_growth")[dest]


@pytest.mark.skipif(FAST_RUN, reason="slow test")
def test_bfs_bitmask_lrx_10():
    graph = prepare_graph("lrx", n=10)
    assert bfs_bitmask(graph) == load_dataset("lrx_cayley_growth")["10"]


@pytest.mark.skipif(FAST_RUN, reason="slow test")
def test_bfs_bitmask_pancake_9():
    graph = prepare_graph("pancake", n=9)
    assert bfs_bitmask(graph) == load_dataset("pancake_cayley_growth")["9"]
