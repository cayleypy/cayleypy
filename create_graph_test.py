import numpy as np

from cayleypy import create_graph, PermutationGroups


def test_create_graph_from_name():
    graph = create_graph(name="pancake", n=10)
    assert graph.definition == PermutationGroups.pancake(10)


def test_create_graph_permutations():
    graph = create_graph(generators_permutations=[[1, 2, 3, 0], [1, 0, 2, 3]])
    assert graph.definition.generators_permutations == PermutationGroups.lx(4).generators_permutations


def test_create_graph_from_matrices():
    m1 = [[1, 2], [3, 4]]
    m2 = np.array([[5, 6], [7, 8]])
    graph = create_graph(generators_matrices=[m1, m2])
    assert graph.definition.is_matrix_group()
    assert graph.definition.n_generators == 2
    assert np.array_equal(graph.definition.generators_matrices[0].matrix, m1)
    assert np.array_equal(graph.definition.generators_matrices[1].matrix, m2)


def test_create_graph_make_inverse_closed():
    graph = create_graph(name="lx", n=4, make_inverse_closed=True)
    assert graph.definition.name == "lx-4-ic"
    assert graph.definition.n_generators == 3
    assert graph.definition.generators_permutations == [[1, 2, 3, 0], [1, 0, 2, 3], [3, 0, 1, 2]]


def test_create_graph_by_name_and_central_state():
    graph = create_graph(name="lrx", n=8, central_state="00001111")
    assert graph.definition.generators_permutations == PermutationGroups.lrx(8).generators_permutations
    assert graph.definition.central_state == [0, 0, 0, 0, 1, 1, 1, 1]
