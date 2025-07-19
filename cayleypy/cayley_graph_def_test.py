import pytest

from cayleypy import CayleyGraphDef, MatrixGenerator, PermutationGroups


def test_inverse_permutations():
    graph_def = CayleyGraphDef.create([[1, 0, 2, 3], [1, 2, 3, 0], [0, 2, 3, 1]])
    inv = graph_def.with_inverted_generators()
    assert inv.is_permutation_group()
    assert inv.generators_permutations == [[1, 0, 2, 3], [3, 0, 1, 2], [0, 3, 1, 2]]


@pytest.mark.parametrize("modulo", [0, 10, 17])
def test_inverse_matrices(modulo: int):
    x = MatrixGenerator.create([[1, 1, 0], [0, 1, 0], [0, 0, 1]], modulo=modulo)
    x_inv = MatrixGenerator.create([[1, -1, 0], [0, 1, 0], [0, 0, 1]], modulo=modulo)
    graph_def = CayleyGraphDef.for_matrix_group(generators=[x])
    inv = graph_def.with_inverted_generators()
    assert inv.is_matrix_group()
    assert len(inv.generators_matrices) == 1
    assert inv.generators_matrices[0] == x_inv


def test_make_inverse_closed():
    graph = PermutationGroups.lrx(4)
    assert graph.make_inverse_closed() == graph

    graph = PermutationGroups.lx(4).make_inverse_closed()
    assert graph.generators_permutations == [[1, 2, 3, 0], [1, 0, 2, 3], [3, 0, 1, 2]]
    assert graph.generator_names == ["L", "X", "L'"]
