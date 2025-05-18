"""Sanity checks for datasets."""
import math

from cayleypy import load_dataset, CayleyGraph, prepare_graph


def _verify_layers_fast(graph: CayleyGraph, layer_sizes: list[int]):
    if max(layer_sizes) < 100:
        assert layer_sizes == graph.bfs().layer_sizes
    else:
        first_layers = graph.bfs(max_layer_size_to_explore=100).layer_sizes
        assert first_layers == layer_sizes[:len(first_layers)]


# LRX Cayley graphs contain all permutations.
def test_lrx_cayley_growth():
    for key, layer_sizes in load_dataset("lrx_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(CayleyGraph(prepare_graph("lrx", n=n).generators), layer_sizes)


# TopSpin Cayley graphs contain all permutations for even n>=6, and half of all permutations for odd n>=7.
def test_top_spin_cayley_growth():
    for key, layer_sizes in load_dataset("top_spin_cayley_growth").items():
        n = int(key)
        if n % 2 == 0 and n >= 6:
            assert sum(layer_sizes) == math.factorial(n)
        if n % 2 == 1 and n >= 7:
            assert sum(layer_sizes) == math.factorial(n) // 2
        _verify_layers_fast(CayleyGraph(prepare_graph("top_spin", n=n).generators), layer_sizes)


# Number of elements in coset graph for LRX and binary strings is binomial coefficient.
def test_lrx_coset_growth():
    for initial_state, layer_sizes in load_dataset("lrx_coset_growth").items():
        n = len(initial_state)
        k = initial_state.count('1')
        assert sum(layer_sizes) == math.comb(n, k)
        _verify_layers_fast(CayleyGraph(prepare_graph("lrx", n=n).generators, dest=initial_state), layer_sizes)


# Number of elements in coset graph for TopSpin and binary strings is binomial coefficient, for n>=6.
def test_top_spin_coset_growth():
    for initial_state, layer_sizes in load_dataset("top_spin_coset_growth").items():
        n = len(initial_state)
        k = initial_state.count('1')
        if n >= 6:
            assert sum(layer_sizes) == math.comb(n, k)
        _verify_layers_fast(CayleyGraph(prepare_graph("top_spin", n=n).generators, dest=initial_state), layer_sizes)
