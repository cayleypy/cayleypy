import numpy as np

from cayleypy import MatrixGroups
from cayleypy.graphs_lib import PermutationGroups


def test_lrx():
    graph = PermutationGroups.lrx(4)
    assert np.array_equal(graph.generators, [[1, 2, 3, 0], [3, 0, 1, 2], [1, 0, 2, 3]])
    assert graph.generator_names == ["L", "R", "X"]
    assert graph.name == "lrx-4"

    graph = PermutationGroups.lrx(5, k=3)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [3, 1, 2, 0, 4]])
    assert graph.generator_names == ["L", "R", "X"]
    assert graph.name == "lrx-5(k=3)"


def test_top_spin():
    graph = PermutationGroups.top_spin(5)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [3, 2, 1, 0, 4]])

    graph = PermutationGroups.top_spin(5, k=3)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [2, 1, 0, 3, 4]])


def test_all_transpositions():
    graph = PermutationGroups.all_transpositions(3)
    assert np.array_equal(graph.generators, [[1, 0, 2], [2, 1, 0], [0, 2, 1]])
    assert graph.generator_names == ["(0,1)", "(0,2)", "(1,2)"]

    graph = PermutationGroups.all_transpositions(20)
    assert graph.n_generators == (20 * 19) // 2


def test_pancake():
    graph = PermutationGroups.pancake(6)
    assert graph.n_generators == 5
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5"]
    assert np.array_equal(
        graph.generators,
        [[1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [3, 2, 1, 0, 4, 5], [4, 3, 2, 1, 0, 5], [5, 4, 3, 2, 1, 0]],
    )


def test_cubic_pancake():
    graph = PermutationGroups.cubic_pancake(n=15, subset=1)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R2"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
    )

    graph = PermutationGroups.cubic_pancake(n=15, subset=2)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R3"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
    )

    graph = PermutationGroups.cubic_pancake(n=15, subset=3)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R13"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
        ],
    )

    graph = PermutationGroups.cubic_pancake(n=15, subset=4)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R12"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 12, 13, 14],
        ],
    )

    graph = PermutationGroups.cubic_pancake(n=15, subset=5)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R13", "R2"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
    )

    graph = PermutationGroups.cubic_pancake(n=15, subset=6)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R13", "R3"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
            [2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
    )

    graph = PermutationGroups.cubic_pancake(n=15, subset=7)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R13", "R12"]
    assert np.array_equal(
        graph.generators,
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 12, 13, 14],
        ],
    )


def test_burnt_pancake():
    graph = PermutationGroups.burnt_pancake(6)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert np.array_equal(
        graph.generators,
        [
            [6, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11],
            [7, 6, 2, 3, 4, 5, 1, 0, 8, 9, 10, 11],
            [8, 7, 6, 3, 4, 5, 2, 1, 0, 9, 10, 11],
            [9, 8, 7, 6, 4, 5, 3, 2, 1, 0, 10, 11],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11],
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ],
    )


def test_full_reversals():
    graph = graph = PermutationGroups.full_reversals(4)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R[0..1]", "R[0..2]", "R[0..3]", "R[1..2]", "R[1..3]", "R[2..3]"]
    assert np.array_equal(
        graph.generators, [[1, 0, 2, 3], [2, 1, 0, 3], [3, 2, 1, 0], [0, 2, 1, 3], [0, 3, 2, 1], [0, 1, 3, 2]]
    )


def test_signed_reversals():
    graph = graph = PermutationGroups.signed_reversals(3)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R[0..0]", "R[0..1]", "R[0..2]", "R[1..1]", "R[1..2]", "R[2..2]"]
    assert np.array_equal(
        graph.generators,
        [
            [3, 1, 2, 0, 4, 5],
            [4, 3, 2, 1, 0, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 4, 2, 3, 1, 5],
            [0, 5, 4, 3, 2, 1],
            [0, 1, 5, 3, 4, 2],
        ],
    )


def test_cyclic_coxeter():
    graph = PermutationGroups.cyclic_coxeter(4)
    assert graph.n_generators == 4
    assert graph.generator_names == ["(0,1)", "(1,2)", "(2,3)", "(0,3)"]
    assert np.array_equal(graph.generators, [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2], [3, 1, 2, 0]])

    graph = PermutationGroups.cyclic_coxeter(3)
    assert graph.n_generators == 3
    assert np.array_equal(graph.generators, [[1, 0, 2], [0, 2, 1], [2, 1, 0]])


def test_three_cycles():
    graph = PermutationGroups.three_cycles(4)
    assert graph.n_generators == 8
    expected_generators = [
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    assert np.array_equal(graph.generators, expected_generators)


def test_three_cycles_0ij():
    graph = PermutationGroups.three_cycles_0ij(4)
    assert graph.n_generators == 6
    expected_generators = [[1, 2, 0, 3], [1, 3, 2, 0], [2, 0, 1, 3], [2, 1, 3, 0], [3, 0, 2, 1], [3, 1, 0, 2]]
    assert np.array_equal(graph.generators, expected_generators)


def test_derangements():
    assert PermutationGroups.derangements(2).generators == [[1, 0]]
    assert PermutationGroups.derangements(3).generators == [[1, 2, 0], [2, 0, 1]]
    assert len(PermutationGroups.derangements(4).generators) == 9
    assert len(PermutationGroups.derangements(5).generators) == 44


def test_rapaport_m1():
    graph_n4 = PermutationGroups.rapaport_m1(4)
    assert graph_n4.generators == [[1, 0, 2, 3], [1, 0, 3, 2], [0, 2, 1, 3]]
    graph_n5 = PermutationGroups.rapaport_m1(5)
    assert graph_n5.generators == [[1, 0, 2, 3, 4], [1, 0, 3, 2, 4], [0, 2, 1, 3, 4], [0, 2, 1, 4, 3]]
    graph_n6 = PermutationGroups.rapaport_m1(6)
    assert graph_n6.generators == [
        [1, 0, 2, 3, 4, 5],
        [1, 0, 3, 2, 4, 5],
        [1, 0, 3, 2, 5, 4],
        [0, 2, 1, 3, 4, 5],
        [0, 2, 1, 4, 3, 5],
    ]


def test_rapaport_m2():
    graph_n5 = PermutationGroups.rapaport_m2(5)
    assert graph_n5.generators == [[1, 0, 2, 3, 4], [1, 0, 3, 2, 4], [0, 2, 1, 4, 3]]
    graph_n6 = PermutationGroups.rapaport_m2(6)
    assert graph_n6.generators == [[1, 0, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4], [0, 2, 1, 4, 3, 5]]


def test_all_cycles():
    graph = PermutationGroups.all_cycles(3)
    assert graph.n_generators == 5
    expected = [
        [1, 0, 2],  # (0 1)
        [2, 1, 0],  # (0 2)
        [0, 2, 1],  # (1 2)
        [1, 2, 0],  # (0 1 2)
        [2, 0, 1],  # (0 2 1)
    ]
    for gen in expected:
        assert gen in graph.generators

    # https://oeis.org/A006231
    assert PermutationGroups.all_cycles(4).n_generators == 20
    assert PermutationGroups.all_cycles(5).n_generators == 84
    assert PermutationGroups.all_cycles(6).n_generators == 409


def test_wrapped_k_cycles():
    graph = PermutationGroups.wrapped_k_cycles(5, 3)
    assert graph.generators == [[1, 2, 0, 3, 4], [0, 2, 3, 1, 4], [0, 1, 3, 4, 2], [3, 1, 2, 4, 0], [1, 4, 2, 3, 0]]


def test_heisenberg():
    graph1 = MatrixGroups.heisenberg()
    assert graph1.name == "heisenberg"
    assert graph1.n_generators == 4
    assert graph1.generators_inverse_closed

    graph2 = MatrixGroups.heisenberg(modulo=10)
    assert graph2.name == "heisenberg%10"
    assert graph2.n_generators == 4
    assert graph1.generators_inverse_closed
