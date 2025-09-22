from math import comb

import numpy as np

from cayleypy import MatrixGroups
from cayleypy.cayley_graph_def import MatrixGenerator
from cayleypy.graphs_lib import PermutationGroups
from cayleypy.permutation_utils import permutation_from_cycles


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


def test_transposons():
    graph = PermutationGroups.transposons(4)
    assert np.array_equal(
        graph.generators,
        [
            [1, 0, 2, 3],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [2, 0, 1, 3],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 1, 3, 2],
        ],
    )
    assert graph.generator_names == [
        "T[0..0,1]",
        "T[0..0,2]",
        "T[0..0,3]",
        "T[0..1,2]",
        "T[0..1,3]",
        "T[0..2,3]",
        "T[1..1,2]",
        "T[1..1,3]",
        "T[1..2,3]",
        "T[2..2,3]",
    ]


def test_block_interchange():
    graph = PermutationGroups.block_interchange(4)
    assert np.array_equal(
        graph.generators,
        [
            [1, 0, 2, 3],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [2, 1, 0, 3],
            [2, 3, 1, 0],
            [3, 1, 2, 0],
            [2, 0, 1, 3],
            [2, 3, 0, 1],
            [3, 2, 0, 1],
            [3, 0, 1, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 2, 1],
            [0, 3, 1, 2],
            [0, 1, 3, 2],
        ],
    )
    assert graph.generator_names == [
        "I[0..0,1..1]",
        "I[0..0,1..2]",
        "I[0..0,1..3]",
        "I[0..0,2..2]",
        "I[0..0,2..3]",
        "I[0..0,3..3]",
        "I[0..1,2..2]",
        "I[0..1,2..3]",
        "I[0..1,3..3]",
        "I[0..2,3..3]",
        "I[1..1,2..2]",
        "I[1..1,2..3]",
        "I[1..1,3..3]",
        "I[1..2,3..3]",
        "I[2..2,3..3]",
    ]


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


def test_three_cycles_01i():
    graph = PermutationGroups.three_cycles_01i(4)
    assert graph.n_generators == 4
    assert graph.generators_inverse_closed
    expected_generators = [[1, 2, 0, 3], [2, 0, 1, 3], [1, 3, 2, 0], [3, 0, 2, 1]]
    assert np.array_equal(graph.generators, expected_generators)
    assert graph.generator_names == ["(0 1 2)", "(1 0 2)", "(0 1 3)", "(1 0 3)"]

    graph = PermutationGroups.three_cycles_01i(4, add_inverses=False)
    assert graph.n_generators == 2
    assert not graph.generators_inverse_closed
    expected_generators = [[1, 2, 0, 3], [1, 3, 2, 0]]
    assert np.array_equal(graph.generators, expected_generators)
    assert graph.generator_names == ["(0 1 2)", "(0 1 3)"]


def test_stars():
    assert len(PermutationGroups.stars(3).generators) == 2
    assert len(PermutationGroups.stars(4).generators) == 3
    assert len(PermutationGroups.stars(5).generators) == 4
    result_3 = PermutationGroups.stars(3)
    assert result_3.generators == [[1, 0, 2], [2, 1, 0]]
    result_4 = PermutationGroups.stars(4)
    assert result_4.generators == [[1, 0, 2, 3], [2, 1, 0, 3], [3, 1, 2, 0]]


def test_generalized_stars():
    assert len(PermutationGroups.generalized_stars(3, 1).generators) == 2
    assert len(PermutationGroups.generalized_stars(4, 2).generators) == 4
    assert len(PermutationGroups.generalized_stars(5, 2).generators) == 6
    result_3 = PermutationGroups.generalized_stars(3, 1)
    assert result_3.generators == [[1, 0, 2], [2, 1, 0]]
    result_4 = PermutationGroups.generalized_stars(4, 2)
    assert result_4.generators == [[2, 1, 0, 3], [3, 1, 2, 0], [0, 2, 1, 3], [0, 3, 2, 1]]


def test_derangements():
    assert PermutationGroups.derangements(2).generators == [[1, 0]]
    assert PermutationGroups.derangements(3).generators == [[1, 2, 0], [2, 0, 1]]
    assert len(PermutationGroups.derangements(4).generators) == 9
    assert len(PermutationGroups.derangements(5).generators) == 44


def test_involutive_derangements():
    assert PermutationGroups.involutive_derangements(2).generators == [[1, 0]]
    assert PermutationGroups.involutive_derangements(4).generators == [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    assert len(PermutationGroups.involutive_derangements(6).generators) == 15
    assert len(PermutationGroups.involutive_derangements(8).generators) == 105


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


def test_larx():
    graph1 = PermutationGroups.larx(3)
    assert graph1.generators == [[1, 0, 2], [0, 2, 1]]
    graph2 = PermutationGroups.larx(4)
    assert graph2.generators == [[1, 0, 2, 3], [0, 2, 3, 1]]


def test_sheveleva2():
    graph = PermutationGroups.sheveleva2(5, 2)
    assert graph.generators == [[1, 0, 2, 3, 4], [0, 2, 3, 4, 1]]
    graph2 = PermutationGroups.sheveleva2(8, 4)
    assert graph2.generators == [[1, 0, 3, 2, 4, 7, 6, 5], [0, 2, 1, 4, 5, 6, 3, 7]]


def test_heisenberg():
    graph1 = MatrixGroups.heisenberg()
    assert graph1.name == "heisenberg-3-ic"
    assert graph1.n_generators == 4
    assert graph1.generator_names == ["x", "y", "x'", "y'"]
    assert graph1.generators_inverse_closed

    graph2 = MatrixGroups.heisenberg(modulo=10)
    assert graph2.name == "heisenberg-3%10-ic"
    assert graph2.n_generators == 4
    assert graph2.generator_names == ["x", "y", "x'", "y'"]
    assert graph2.generators_inverse_closed

    graph3 = MatrixGroups.heisenberg(add_inverses=False)
    assert graph3.name == "heisenberg-3"
    assert graph3.n_generators == 2
    assert graph3.generator_names == ["x", "y"]
    assert np.array_equal(graph3.generators_matrices[0].matrix, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(graph3.generators_matrices[1].matrix, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    assert not graph3.generators_inverse_closed

    graph4 = MatrixGroups.heisenberg(n=5, modulo=100)
    assert graph4.name == "heisenberg-5%100-ic"
    assert graph4.n_generators == 12
    assert graph4.generator_names == ["x1", "x2", "x3", "y1", "y2", "y3", "x1'", "x2'", "x3'", "y1'", "y2'", "y3'"]
    assert np.array_equal(
        graph4.generators_matrices[0].matrix,
        [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
    )
    assert graph4.generators_inverse_closed


def test_sl_fund_roots():
    graph = MatrixGroups.special_linear_fundamental_roots(2)
    assert graph.n_generators == 4
    assert graph.generators_inverse_closed
    assert graph.generators == [
        MatrixGenerator.create([[1, 1], [0, 1]]),
        MatrixGenerator.create([[1, -1], [0, 1]]),
        MatrixGenerator.create([[1, 0], [1, 1]]),
        MatrixGenerator.create([[1, 0], [-1, 1]]),
    ]

    graph = MatrixGroups.special_linear_fundamental_roots(3)
    assert graph.n_generators == 8
    assert graph.generators_inverse_closed
    assert graph.generators == [
        MatrixGenerator.create([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
        MatrixGenerator.create([[1, -1, 0], [0, 1, 0], [0, 0, 1]]),
        MatrixGenerator.create([[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
        MatrixGenerator.create([[1, 0, 0], [-1, 1, 0], [0, 0, 1]]),
        MatrixGenerator.create([[1, 0, 0], [0, 1, 1], [0, 0, 1]]),
        MatrixGenerator.create([[1, 0, 0], [0, 1, -1], [0, 0, 1]]),
        MatrixGenerator.create([[1, 0, 0], [0, 1, 0], [0, 1, 1]]),
        MatrixGenerator.create([[1, 0, 0], [0, 1, 0], [0, -1, 1]]),
    ]


def test_sl_root_weyl():
    graph = MatrixGroups.special_linear_root_weyl(2)
    assert graph.n_generators == 4
    assert graph.generators_inverse_closed
    assert all(np.linalg.det(a.matrix) == 1 for a in graph.generators)
    assert graph.generators == [
        MatrixGenerator.create([[1, 1], [0, 1]]),
        MatrixGenerator.create([[1, -1], [0, 1]]),
        MatrixGenerator.create([[0, 1], [-1, 0]]),
        MatrixGenerator.create([[0, -1], [1, 0]]),
    ]

    graph = MatrixGroups.special_linear_root_weyl(3)
    assert graph.n_generators == 4
    assert graph.generators_inverse_closed
    assert all(np.linalg.det(a.matrix) == 1 for a in graph.generators)
    assert graph.generators == [
        MatrixGenerator.create([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
        MatrixGenerator.create([[1, -1, 0], [0, 1, 0], [0, 0, 1]]),
        MatrixGenerator.create([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        MatrixGenerator.create([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    ]

    graph = MatrixGroups.special_linear_root_weyl(4)
    assert graph.n_generators == 4
    assert graph.generators_inverse_closed
    assert all(np.linalg.det(a.matrix) == 1 for a in graph.generators)
    assert graph.generators == [
        MatrixGenerator.create([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        MatrixGenerator.create([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        MatrixGenerator.create([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0]]),
        MatrixGenerator.create([[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
    ]


def test_increasing_k_cycles_basic():
    g = PermutationGroups.increasing_k_cycles(3, 2)
    assert g.generators_permutations == [[1, 0, 2], [2, 1, 0], [0, 2, 1]]

    n, k = 5, 3
    g = PermutationGroups.increasing_k_cycles(n, k)
    assert g.n_generators == comb(n, k)
    expected_1 = permutation_from_cycles(n, [[0, 1, 2]])
    expected_2 = permutation_from_cycles(n, [[1, 3, 4]])
    assert expected_1 in g.generators_permutations
    assert expected_2 in g.generators_permutations

    unexpected_inv = permutation_from_cycles(n, [[0, 2, 1]])
    assert unexpected_inv not in g.generators_permutations

    n, k = 4, 2
    g = PermutationGroups.increasing_k_cycles(n, k)
    assert g.n_generators == comb(n, k)


def test_consecutive_k_cycles_basic_counts():
    g = PermutationGroups.consecutive_k_cycles(5, 2)
    assert g.n_generators == (5 - 2 + 1)

    g3 = PermutationGroups.consecutive_k_cycles(6, 3)
    assert g3.n_generators == (6 - 3 + 1)


def test_consecutive_k_cycles_contains_expected_generators():
    g = PermutationGroups.consecutive_k_cycles(3, 2)
    assert g.generators_permutations == [[1, 0, 2], [0, 2, 1]]

    n, k = 6, 3
    g = PermutationGroups.consecutive_k_cycles(n, k)
    fwd = permutation_from_cycles(n, [[0, 1, 2]])
    assert fwd in g.generators_permutations
    fwd2 = permutation_from_cycles(n, [[2, 3, 4]])
    assert fwd2 in g.generators_permutations

    inv = permutation_from_cycles(n, [[0, 2, 1]])
    assert inv not in g.generators_permutations
    assert g.name == "consecutive_k_cycles-6-3"


def test_conjugacy_class():
    n = 4
    classes = {(2, 2): None, (3,): None}
    graph = PermutationGroups.conjugacy_classes(n, classes=classes)
    assert np.array_equal(
        graph.generators,
        [
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [1, 2, 0, 3],
            [2, 0, 1, 3],
            [1, 3, 2, 0],
            [3, 0, 2, 1],
            [2, 1, 3, 0],
            [3, 1, 0, 2],
        ],
    )
    assert graph.generator_names == [
        "(2,2)_1",
        "(2,2)_2",
        "(2,2)_3",
        "(3,1)_1",
        "(3,1)_2",
        "(3,1)_3",
        "(3,1)_4",
        "(3,1)_5",
        "(3,1)_6",
        "(3,1)_7",
        "(3,1)_8",
    ]
    assert graph.name == "conjugacy_class-4-2,2-3,1"

    n = 5
    classes = {(2, 2): 5, (3,): 7}
    graph = PermutationGroups.conjugacy_classes(n, classes=classes)
    assert graph.generator_names == [
        "(2,2,1)_1",
        "(2,2,1)_2",
        "(2,2,1)_3",
        "(2,2,1)_4",
        "(2,2,1)_5",
        "(3,1,1)_1",
        "(3,1,1)_2",
        "(3,1,1)_3",
        "(3,1,1)_4",
        "(3,1,1)_5",
        "(3,1,1)_6",
        "(3,1,1)_7",
    ]
    assert graph.name == "conjugacy_class-5-2,2,1_5-3,1,1_7"


def test_rand_generators():
    n = 6
    k = 4
    graph = PermutationGroups.rand_generators(n, k)
    assert len(graph.generator_names) == k
    assert graph.name == "rand_generators-6-4"
