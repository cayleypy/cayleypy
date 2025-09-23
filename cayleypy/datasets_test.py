"""Sanity checks for datasets."""

import math
import os

import pytest

from . import GapPuzzles
from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef
from .datasets import load_dataset
from .graphs_lib import PermutationGroups, MatrixGroups
from .puzzles.puzzles import Puzzles

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


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
        _verify_layers_fast(PermutationGroups.lrx(n), layer_sizes)


def test_lx_cayley_growth():
    # See https://oeis.org/A039745
    oeis_a039745 = [None, 0, 1, 2, 6, 11, 18, 25, 35, 45, 58, 71, 87, 103, 122, 141]
    for key, layer_sizes in load_dataset("lx_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) - 1 == oeis_a039745[n]
        _verify_layers_fast(PermutationGroups.lx(n), layer_sizes)


def test_burnt_pancake_cayley_growth():
    oeis_a078941 = [None, 1, 4, 6, 8, 10, 12, 14, 15, 17, 18, 19, 21]
    for key, layer_sizes in load_dataset("burnt_pancake_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n) * 2**n
        assert len(layer_sizes) - 1 == oeis_a078941[n]
        _verify_layers_fast(PermutationGroups.burnt_pancake(n), layer_sizes)


# TopSpin Cayley graphs contain all permutations for even n>=6, and half of all permutations for odd n>=7.
def test_top_spin_cayley_growth():
    for key, layer_sizes in load_dataset("top_spin_cayley_growth").items():
        n = int(key)
        if n % 2 == 0 and n >= 6:
            assert sum(layer_sizes) == math.factorial(n)
        if n % 2 == 1 and n >= 7:
            assert sum(layer_sizes) == math.factorial(n) // 2
        _verify_layers_fast(PermutationGroups.top_spin(n), layer_sizes)


def test_all_transpositions_cayley_growth():
    for key, layer_sizes in load_dataset("all_transpositions_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) == n  # Graph diameter is n-1.
        assert layer_sizes[-1] == math.factorial(n - 1)  # Size of last layer is (n-1)!.


def test_transposons_cayley_growth():
    oeis_a065603 = [None, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9]
    for key, layer_sizes in load_dataset("transposons_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) - 1 == oeis_a065603[n]


def test_block_interchange_cayley_growth():
    for key, layer_sizes in load_dataset("block_interchange_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) == n // 2 + 1


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
        _verify_layers_fast(PermutationGroups.pancake(n), layer_sizes)


def test_full_reversals_cayley_growth():
    for key, layer_sizes in load_dataset("full_reversals_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(PermutationGroups.full_reversals(n), layer_sizes)
        assert len(layer_sizes) == n  # Graph diameter is n-1.
        if n >= 3:
            assert layer_sizes[-1] == 2  # Size of last layer is 2.


def test_signed_reversals_cayley_growth():
    for key, layer_sizes in load_dataset("signed_reversals_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n) * 2**n
        _verify_layers_fast(PermutationGroups.signed_reversals(n), layer_sizes)


# Number of elements in coset graph for LRX and binary strings is binomial coefficient.
@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_lrx_coset_growth():
    for central_state, layer_sizes in load_dataset("lrx_coset_growth").items():
        n = len(central_state)
        k = central_state.count("1")
        assert sum(layer_sizes) == math.comb(n, k)
        graph = PermutationGroups.lrx(n).with_central_state(central_state)
        _verify_layers_fast(graph, layer_sizes, max_layer_size=100)


# Number of elements in coset graph for TopSpin and binary strings is binomial coefficient, for n>=6.
@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_top_spin_coset_growth():
    for central_state, layer_sizes in load_dataset("top_spin_coset_growth").items():
        n = len(central_state)
        k = central_state.count("1")
        if n >= 6:
            assert sum(layer_sizes) == math.comb(n, k)
        graph = PermutationGroups.top_spin(n).with_central_state(central_state)
        _verify_layers_fast(graph, layer_sizes, max_layer_size=100)


def test_coxeter_cayley_growth():
    for key, layer_sizes in load_dataset("coxeter_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert len(layer_sizes) - 1 == n * (n - 1) // 2
        assert layer_sizes == layer_sizes[::-1]  # Growth function is a palindrome.


def test_larx_cayley_growth():
    for key, layer_sizes in load_dataset("larx_cayley_growth").items():
        n = int(key)
        _verify_layers_fast(PermutationGroups.larx(n), layer_sizes)


# This test checks that the hash function is good when states are bit-encoded (and there are no collisions).
@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_coxeter_cayley_growth_upto_200000():
    for key, layer_sizes in load_dataset("coxeter_cayley_growth").items():
        n = int(key)
        _verify_layers_fast(PermutationGroups.coxeter(n), layer_sizes, max_layer_size=200000)


def test_cyclic_coxeter_cayley_growth():
    for key, layer_sizes in load_dataset("cyclic_coxeter_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(PermutationGroups.cyclic_coxeter(n), layer_sizes)


def test_rapaport_m1_cayley_growth():
    for key, layer_sizes in load_dataset("rapaport_m1_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(PermutationGroups.rapaport_m1(n), layer_sizes)


def test_rapaport_m2_cayley_growth():
    for key, layer_sizes in load_dataset("rapaport_m2_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(PermutationGroups.rapaport_m2(n), layer_sizes)


def test_wrapped_k_cycles_cayley_growth():
    for key, layer_sizes in load_dataset("wrapped_k_cycles_cayley_growth").items():
        n, k = map(int, key.split(","))
        _verify_layers_fast(PermutationGroups.wrapped_k_cycles(n, k), layer_sizes)


def test_stars_cayley_growth():
    for key, layer_sizes in load_dataset("stars_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(PermutationGroups.stars(n), layer_sizes)


def test_derangements_cayley_growth():
    for key, layer_sizes in load_dataset("derangements_cayley_growth").items():
        n = int(key)
        if n >= 4:
            assert sum(layer_sizes) == math.factorial(n)
        _verify_layers_fast(PermutationGroups.derangements(n), layer_sizes)


def test_involutive_derangements_cayley_growth():
    for key, layer_sizes in load_dataset("involutive_derangements_cayley_growth").items():
        n = int(key)
        _verify_layers_fast(PermutationGroups.involutive_derangements(n), layer_sizes)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_hungarian_rings_growth():
    for key, layer_sizes in load_dataset("hungarian_rings_growth").items():
        parameters = list(map(int, key.split(",")))
        assert len(parameters) == 4
        (left_size, left_index, right_size, right_index) = parameters
        n = left_size + right_size - (2 if left_index > 0 and right_index > 0 else 1)
        if n < 6 or n > 12:
            continue
        _verify_layers_fast(Puzzles.hungarian_rings(*parameters), layer_sizes)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_heisenberg_growth():
    for key, layer_sizes in load_dataset("heisenberg_growth").items():
        n, modulo = map(int, key.split(","))
        assert sum(layer_sizes) == modulo ** (2 * n - 3)
        _verify_layers_fast(MatrixGroups.heisenberg(n=n, modulo=modulo), layer_sizes)


def test_sl_fund_roots_growth():
    for n in [2, 3]:
        for key, layer_sizes in load_dataset(f"sl_{n}_fund_roots_growth").items():
            m = int(key)
            if m in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
                assert sum(layer_sizes) == math.prod(m**n - m**i for i in range(n)) / (m - 1)
            _verify_layers_fast(MatrixGroups.special_linear_fundamental_roots(n, modulo=m), layer_sizes)


def test_sl_root_weyl_growth():
    for n in [2, 3]:
        for key, layer_sizes in load_dataset(f"sl_{n}_root_weyl_growth").items():
            m = int(key)
            if m in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
                assert sum(layer_sizes) == math.prod(m**n - m**i for i in range(n)) / (m - 1)
            _verify_layers_fast(MatrixGroups.special_linear_root_weyl(n, modulo=m), layer_sizes)


def test_puzzles_growth():
    data = load_dataset("puzzles_growth")
    _verify_layers_fast(Puzzles.rubik_cube(3, metric="HTM"), data["cube_333_htm"])
    _verify_layers_fast(Puzzles.rubik_cube(3, metric="QTM"), data["cube_333_qtm"])
    _verify_layers_fast(Puzzles.mini_pyramorphix(), data["mini_pyramorphix"])
    _verify_layers_fast(Puzzles.pyraminx(), data["pyraminx"])
    _verify_layers_fast(Puzzles.starminx(), data["starminx"])
    _verify_layers_fast(Puzzles.starminx_2(), data["starminx_2"])


def test_puzzles_growth_cube_2x2x2():
    data = load_dataset("puzzles_growth")
    _verify_layers_fast(Puzzles.rubik_cube(2, metric="fixed_HTM"), data["cube_222_fixed_htm"])
    _verify_layers_fast(Puzzles.rubik_cube(2, metric="fixed_QTM"), data["cube_222_fixed_qtm"])
    _verify_layers_fast(Puzzles.rubik_cube(2, metric="QSTM"), data["cube_222_qstm"])
    _verify_layers_fast(Puzzles.rubik_cube(2, metric="HTM"), data["cube_222_htm"])
    _verify_layers_fast(Puzzles.rubik_cube(2, metric="QTM"), data["cube_222_qtm"])
    _verify_layers_fast(Puzzles.rubik_cube(2, metric="ATM"), data["cube_222_atm"])

    for name in ["cube_222_fixed_htm", "cube_222_fixed_qtm"]:
        assert sum(data[name]) == 3674160
    for name in ["cube_222_qstm", "cube_222_htm", "cube_222_qtm", "cube_222_atm"]:
        assert sum(data[name]) == 88179840


def test_gap_puzzles_growth_cubes():
    data = load_dataset("puzzles_growth")
    _verify_layers_fast(GapPuzzles.puzzle("2x2x2"), data["cube_222_qstm"])
    _verify_layers_fast(GapPuzzles.puzzle("3x3x3"), data["cube_333_qtm"])


@pytest.mark.parametrize(
    "puzzle_name",
    [
        "dino",
        "master_pyramorphix",
        "mastermorphix",
        "pyraminx",
        "pyramorphix",
        "skewb_diamond",
        "starminx",
        "starminx_2",
        "tetraminx",
    ],
)
def test_gap_puzzles_growth(puzzle_name: str):
    _verify_layers_fast(GapPuzzles.puzzle(puzzle_name), load_dataset("puzzles_growth")[puzzle_name])


def test_globes_growth():
    for key, layer_sizes in load_dataset("globes_growth").items():
        a, b = map(int, key.split(","))
        _verify_layers_fast(Puzzles.globe_puzzle(a, b), layer_sizes)


def test_all_cycles_cayley_growth():
    for key, layer_sizes in load_dataset("all_cycles_cayley_growth").items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        assert all(isinstance(x, int) and x >= 0 for x in layer_sizes)
        if n <= 5:
            _verify_layers_fast(PermutationGroups.all_cycles(n), layer_sizes)


def test_increasing_k_cycles_cayley_growth():
    data = load_dataset("increasing_k_cycles_cayley_growth")
    for key, layer_sizes in data.items():
        n, k = map(int, key.split(","))

        if n == k and k > 2:
            continue

        if k % 2 == 0:
            assert sum(layer_sizes) == math.factorial(n)
        else:
            if n >= 3:
                assert sum(layer_sizes) == math.factorial(n) // 2

        graph_def = PermutationGroups.increasing_k_cycles(n, k)
        first_layers = CayleyGraph(graph_def).bfs(max_layer_size_to_explore=2000).layer_sizes
        assert first_layers == layer_sizes[: len(first_layers)]


def test_consecutive_k_cycles_cayley_growth():
    data = load_dataset("consecutive_k_cycles_cayley_growth")
    for key, layer_sizes in data.items():
        n, k = map(int, key.split(","))
        if k == n and n > 2:
            assert sum(layer_sizes) == n
        elif k % 2 == 0:
            assert sum(layer_sizes) == math.factorial(n)
        else:
            if n >= 3:
                assert sum(layer_sizes) == math.factorial(n) // 2
        graph_def = PermutationGroups.consecutive_k_cycles(n, k)
        first_layers = CayleyGraph(graph_def).bfs(max_layer_size_to_explore=2000).layer_sizes
        assert first_layers == layer_sizes[: len(first_layers)]


def test_down_cycles_cayley_growth():
    data = load_dataset("down_cycles_cayley_growth")
    for key, layer_sizes in data.items():
        n = int(key)
        assert sum(layer_sizes) == math.factorial(n)
        graph_def = PermutationGroups.down_cycles(n)
        first_layers = CayleyGraph(graph_def).bfs(max_layer_size_to_explore=2000).layer_sizes
        assert first_layers == layer_sizes[: len(first_layers)]
        assert all(isinstance(x, int) and x >= 0 for x in layer_sizes)
        if n <= 6:
            _verify_layers_fast(PermutationGroups.down_cycles(n), layer_sizes)
