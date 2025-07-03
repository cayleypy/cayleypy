import numpy as np

from cayleypy import prepare_graph
from cayleypy.cayley_graph import CayleyGraph
from cayleypy.graphs_lib import MINI_PYRAMORPHIX_ALLOWED_MOVES, PYRAMINX_MOVES, MEGAMINX_MOVES, PermutationGroups
from cayleypy.permutation_utils import inverse_permutation, is_permutation


def test_lrx():
    graph = PermutationGroups.lrx(4)
    assert np.array_equal(graph.generators, [[1, 2, 3, 0], [3, 0, 1, 2], [1, 0, 2, 3]])
    assert graph.generator_names == ["L", "R", "X"]

    graph = PermutationGroups.lrx(5, k=3)
    assert np.array_equal(graph.generators, [[1, 2, 3, 4, 0], [4, 0, 1, 2, 3], [3, 1, 2, 0, 4]])
    assert graph.generator_names == ["L", "R", "X"]


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


def test_cube333():
    graph = prepare_graph("cube_3/3/3_12gensQTM")
    assert graph.n_generators == 12

    graph = prepare_graph("cube_3/3/3_18gensHTM")
    assert graph.n_generators == 18


def test_cyclic_coxeter():
    graph = PermutationGroups.cyclic_coxeter(4)
    assert graph.n_generators == 4
    assert graph.generator_names == ["(0,1)", "(1,2)", "(2,3)", "(0,3)"]
    assert np.array_equal(graph.generators, [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2], [3, 1, 2, 0]])

    graph = PermutationGroups.cyclic_coxeter(3)
    assert graph.n_generators == 3
    assert np.array_equal(graph.generators, [[1, 0, 2], [0, 2, 1], [2, 1, 0]])


def test_mini_pyramorphix():
    graph = prepare_graph("mini_pyramorphix")
    assert graph.n_generators == len(MINI_PYRAMORPHIX_ALLOWED_MOVES)
    assert graph.generator_names == list(MINI_PYRAMORPHIX_ALLOWED_MOVES.keys())
    expected_generators = np.array([MINI_PYRAMORPHIX_ALLOWED_MOVES[k] for k in graph.generator_names])
    assert np.array_equal(graph.generators, expected_generators)
    for gen in graph.generators:
        assert len(gen) == 24
        assert is_permutation(gen)
    identity = list(range(24))
    assert any(gen != identity for gen in graph.generators)
    for gen in graph.generators:
        inverse = inverse_permutation(gen)
        restored = [gen[i] for i in inverse]
        assert restored == list(range(24))
    assert set(graph.generator_names) == set(MINI_PYRAMORPHIX_ALLOWED_MOVES.keys())


def test_pyraminx():
    perm_set_length = 36
    graph = prepare_graph("pyraminx")
    assert graph.n_generators == len(PYRAMINX_MOVES) * 2  # inverse generators are not listed in PYRAMINX_MOVES

    graph_gens = dict(zip(graph.generator_names, graph.generators))
    gen_names = list(PYRAMINX_MOVES.keys())
    gen_names += [x + "_inv" for x in PYRAMINX_MOVES]

    for gen_name, gen in PYRAMINX_MOVES.items():
        assert np.all(graph_gens[gen_name] == gen)
        assert np.all(graph_gens[gen_name + "_inv"] == inverse_permutation(gen))
        assert len(gen) == perm_set_length


def test_megaminx():
    perm_set_length = 120
    graph = prepare_graph("megaminx")
    assert graph.n_generators == len(MEGAMINX_MOVES) * 2  # inverse generators are not listed in MEGAMINX_MOVES

    graph_gens = dict(zip(graph.generator_names, graph.generators))
    gen_names = list(MEGAMINX_MOVES.keys())
    gen_names += [x + "_inv" for x in MEGAMINX_MOVES]

    for gen_name, gen in MEGAMINX_MOVES.items():
        assert np.all(graph_gens[gen_name] == gen)
        assert np.all(graph_gens[gen_name + "_inv"] == inverse_permutation(gen))
        assert len(gen) == perm_set_length

    graph = CayleyGraph(graph, device="cpu")
    assert graph.bfs(max_diameter=4).layer_sizes == [1, 24, 408, 6208, 90144]


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
