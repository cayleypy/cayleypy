"""Sanity checks for datasets."""

import math

from cayleypy import load_dataset, CayleyGraph, CayleyGraphDef, prepare_graph


def _verify_layers_fast(graph_def: CayleyGraphDef, layer_sizes: list[int], max_layer_size=1000):
    graph = CayleyGraph(graph_def)
    if max(layer_sizes) < max_layer_size:
        assert layer_sizes == graph.bfs().layer_sizes
    else:
        first_layers = graph.bfs(max_layer_size_to_explore=max_layer_size).layer_sizes
        assert first_layers == layer_sizes[: len(first_layers)]


# LRX Cayley graphs contain all permutations.
# It's conjectured that for n>=4, diameter of LRX Cayley graph is n(n-1)/2. See https://oeis.org/A186783.
def test_lrx_cayley_growth():
    for key, layer_sizes in load_dataset("lrx_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        if n >= 4:
            assert len(layer_sizes) - 1 == n * (n - 1) // 2
        _verify_layers_fast(prepare_graph("lrx", n=n), layer_sizes)


def test_burnt_pancake_cayley_growth():
    oeis_a078941 = [None, 1, 4, 6, 8, 10, 12, 14, 15, 17, 18, 19, 21]
    for key, layer_sizes in load_dataset("burnt_pancake_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n) * 2**n
        assert len(layer_sizes) - 1 == oeis_a078941[n]
        _verify_layers_fast(prepare_graph("burnt_pancake", n=n), layer_sizes)


# TopSpin Cayley graphs contain all permutations for even n>=6, and half of all permutations for odd n>=7.
def test_top_spin_cayley_growth():
    for key, layer_sizes in load_dataset("top_spin_cayley_growth").items():
        n = int(key)
        if n % 2 == 0 and n >= 6:
            assert sum(layer_sizes) == math.factorial(n)
        if n % 2 == 1 and n >= 7:
            assert sum(layer_sizes) == math.factorial(n) // 2
        _verify_layers_fast(prepare_graph("top_spin", n=n), layer_sizes)


def test_all_transpositions_cayley_growth():
    for key, layer_sizes in load_dataset("all_transpositions_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) == n  # Graph diameter is n-1.
        assert layer_sizes[-1] == math.factorial(n - 1)  # Size of last layer is (n-1)!.


def test_pancake_cayley_growth():
    # See https://oeis.org/A058986
    oeis_a058986 = [None, 0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    # See https://oeis.org/A067607
    oeis_a067607 = [None, 1, 1, 1, 3, 20, 2, 35, 455, 5804, 73232, 6, 167, 2001, 24974, 339220, 4646117, 65758725]
    for key, layer_sizes in load_dataset("pancake_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) - 1 == oeis_a058986[n]
        assert layer_sizes[-1] == oeis_a067607[n]
        _verify_layers_fast(prepare_graph("pancake", n=n), layer_sizes)


def test_full_reversals_cayley_growth():
    for key, layer_sizes in load_dataset("full_reversals_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(prepare_graph("full_reversals", n=n), layer_sizes)
        assert len(layer_sizes) == n  # Graph diameter is n-1.
        if n >= 3:
            assert layer_sizes[-1] == 2  # Size of last layer is 2.


# Number of elements in coset graph for LRX and binary strings is binomial coefficient.
def test_lrx_coset_growth():
    for central_state, layer_sizes in load_dataset("lrx_coset_growth").items():
        n = len(central_state)
        k = central_state.count("1")
        assert sum(layer_sizes) == math.comb(n, k)
        graph = prepare_graph("lrx", n=n).with_central_state(central_state)
        _verify_layers_fast(graph, layer_sizes, max_layer_size=100)


# Number of elements in coset graph for TopSpin and binary strings is binomial coefficient, for n>=6.
def test_top_spin_coset_growth():
    for central_state, layer_sizes in load_dataset("top_spin_coset_growth").items():
        n = len(central_state)
        k = central_state.count("1")
        if n >= 6:
            assert sum(layer_sizes) == math.comb(n, k)
        graph = prepare_graph("top_spin", n=n).with_central_state(central_state)
        _verify_layers_fast(graph, layer_sizes, max_layer_size=100)


def test_coxeter_cayley_growth():
    for key, layer_sizes in load_dataset("coxeter_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(prepare_graph("coxeter", n=n), layer_sizes)
        assert len(layer_sizes) - 1 == n * (n - 1) // 2


def test_cyclic_coxeter_cayley_growth():
    for key, layer_sizes in load_dataset("cyclic_coxeter_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(prepare_graph("cyclic_coxeter", n=n), layer_sizes)


def test_hungarian_rings_growth():
    for key, layer_sizes in load_dataset("hungarian_rings_growth").items():
        n = int(key)
        assert n % 2 == 0
        ring_size = (n + 2) // 2
        assert sum(layer_sizes) == math.factorial(n) // (2 if (ring_size % 2 > 0) else 1)
        _verify_layers_fast(prepare_graph("hungarian_rings", n=n), layer_sizes)


def test_puzzles_growth():
    data = load_dataset("puzzles_growth")
    _verify_layers_fast(prepare_graph("cube_2/2/2_9gensHTM"), data["cube_222_htm"])
    _verify_layers_fast(prepare_graph("cube_2/2/2_6gensQTM"), data["cube_222_qtm"])
    _verify_layers_fast(prepare_graph("cube_3/3/3_18gensHTM"), data["cube_333_htm"])
    _verify_layers_fast(prepare_graph("cube_3/3/3_12gensQTM"), data["cube_333_qtm"])
    _verify_layers_fast(prepare_graph("mini_pyramorphix"), data["mini_pyramorphix"])
    _verify_layers_fast(prepare_graph("pyraminx"), data["pyraminx"])
