import numpy as np

from cayleypy import prepare_graph
from cayleypy.permutation_utils import inverse_permutation
from cayleypy.graphs_lib import MINI_PARAMORPHIX_ALLOWED_MOVES


def test_lrx():
    graph = prepare_graph("lrx", n=4)
    assert np.array_equal(graph.generators, [[1, 2, 3, 0], [3, 0, 1, 2], [1, 0, 2, 3]])
    assert graph.generator_names == ["L", "R", "X"]

    graph = prepare_graph("lrx", n=5, k=3)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [3, 1, 2, 0, 4]])
    assert graph.generator_names == ["L", "R", "X"]


def test_top_spin():
    graph = prepare_graph("top_spin", n=5)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [3, 2, 1, 0, 4]])

    graph = prepare_graph("top_spin", n=5, k=3)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [2, 1, 0, 3, 4]])


def test_all_transpositions():
    graph = prepare_graph("all_transpositions", n=3)
    assert np.array_equal(graph.generators, [[1, 0, 2], [2, 1, 0], [0, 2, 1]])
    assert graph.generator_names == ["(0,1)", "(0,2)", "(1,2)"]

    graph = prepare_graph("all_transpositions", n=20)
    assert graph.n_generators == (20 * 19) // 2


def test_pancake():
    graph = prepare_graph("pancake", n=6)
    assert graph.n_generators == 5
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5"]
    assert np.array_equal(graph.generators, [
        [1, 0, 2, 3, 4, 5],
        [2, 1, 0, 3, 4, 5],
        [3, 2, 1, 0, 4, 5],
        [4, 3, 2, 1, 0, 5],
        [5, 4, 3, 2, 1, 0]
    ])


def test_burnt_pancake():
    graph = prepare_graph("burnt_pancake", n=6)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert np.array_equal(graph.generators, [
        [6, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11],
        [7, 6, 2, 3, 4, 5, 1, 0, 8, 9, 10, 11],
        [8, 7, 6, 3, 4, 5, 2, 1, 0, 9, 10, 11],
        [9, 8, 7, 6, 4, 5, 3, 2, 1, 0, 10, 11],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11],
        [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ])


def test_full_reversals():
    graph = prepare_graph("full_reversals", n=4)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R[0..1]", "R[0..2]", "R[0..3]", "R[1..2]", "R[1..3]", "R[2..3]"]
    assert np.array_equal(graph.generators, [
        [1, 0, 2, 3], [2, 1, 0, 3], [3, 2, 1, 0], [0, 2, 1, 3], [0, 3, 2, 1], [0, 1, 3, 2]
    ])


def test_cube333():
    graph = prepare_graph("cube_3/3/3_12gensQTM")
    assert graph.n_generators == 12

    graph = prepare_graph("cube_3/3/3_18gensHTM")
    assert graph.n_generators == 18


def test_cyclic_coxeter():
    graph = prepare_graph("cyclic_coxeter", n=4)
    assert graph.n_generators == 4
    assert graph.generator_names == ["(0,1)", "(1,2)", "(2,3)", "(0,3)"]
    assert np.array_equal(graph.generators, [
        [1, 0, 2, 3],
        [0, 2, 1, 3],
        [0, 1, 3, 2],
        [3, 1, 2, 0]
    ])

    graph = prepare_graph("cyclic_coxeter", n=3)
    assert graph.n_generators == 3
    assert np.array_equal(graph.generators, [
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0]
    ])


def test_mini_paramorphix():
    graph = prepare_graph("mini_paramorphix")
    assert graph.n_generators == len(MINI_PARAMORPHIX_ALLOWED_MOVES)
    assert graph.generator_names == list(MINI_PARAMORPHIX_ALLOWED_MOVES.keys())
    expected_generators = np.array([MINI_PARAMORPHIX_ALLOWED_MOVES[k] for k in graph.generator_names])
    assert np.array_equal(graph.generators, expected_generators)
    for gen in graph.generators:
        assert len(gen) == 24
        assert sorted(gen.tolist()) == list(range(24))
    identity = list(range(24))
    assert any(gen.tolist() != identity for gen in graph.generators)
    for gen in graph.generators:
        inverse = inverse_permutation(gen.tolist())
        restored = [gen[i] for i in inverse]
        assert restored == list(range(24))
    assert set(graph.generator_names) == set(MINI_PARAMORPHIX_ALLOWED_MOVES.keys())
