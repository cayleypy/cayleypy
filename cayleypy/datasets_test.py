"""Sanity checks for datasets."""
import math

from cayleypy import load_dataset


# LRX Cayley graphs contain all permutations.
def test_lrx_cayley_growth():
    for key, layer_sizes in load_dataset("lrx_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)


# TopSpin Cayley graphs contain all permutations for even n>=6, and half of all permutations for odd n>=7.
def test_top_spin_cayley_growth():
    for key, layer_sizes in load_dataset("top_spin_cayley_growth").items():
        n = int(key)
        if n % 2 == 0 and n >= 6:
            assert sum(layer_sizes) == math.factorial(n)
        if n % 2 == 1 and n >= 7:
            assert sum(layer_sizes) == math.factorial(n) // 2


# Number of elements in coset graph for LRX and binary strings is binomial coefficient.
def test_lrx_coset_growth():
    for initial_state, layer_sizes in load_dataset("lrx_coset_growth").items():
        n = len(initial_state)
        k = initial_state.count('1')
        assert sum(layer_sizes) == math.comb(n, k)


# Number of elements in coset graph for TopSpin and binary strings is binomial coefficient, for n>=6.
def test_top_spin_coset_growth():
    for initial_state, layer_sizes in load_dataset("top_spin_coset_growth").items():
        n = len(initial_state)
        k = initial_state.count('1')
        if n >= 6:
            assert sum(layer_sizes) == math.comb(n, k)
