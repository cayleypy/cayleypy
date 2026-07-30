import numpy as np
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


def test_matrix_inv_modular_det_not_one():
    """A6: modular inverse works for matrices with det != 1 (mod m).

    A = [[1, 1], [0, 2]], det=2, modulo=5. gcd(2,5)=1, so invertible mod 5.
    inv(A) = det^{-1} * adj(A) mod 5 = 3 * [[2, -1], [0, 1]] = [[1, 2], [0, 3]] (mod 5).
    """
    a = MatrixGenerator.create([[1, 1], [0, 2]], modulo=5)
    a_inv = a.inv
    eye = np.eye(2, dtype=np.int64)
    assert np.array_equal(a.apply(a_inv.matrix), eye), "A * inv(A) != I mod 5"
    assert np.array_equal(a_inv.apply(a.matrix), eye), "inv(A) * A != I mod 5"
    expected = np.array([[1, 2], [0, 3]], dtype=np.int64)
    assert np.array_equal(a_inv.matrix, expected), f"Expected {expected}, got {a_inv.matrix}"


def test_matrix_inv_modular_non_invertible_raises():
    """A6: matrix with det not coprime to modulo raises ValueError."""
    a = MatrixGenerator.create([[2, 0], [0, 2]], modulo=4)  # det=4, gcd(4,4)=4 != 1
    with pytest.raises(ValueError, match="not invertible"):
        _ = a.inv


def test_make_inverse_closed():
    graph = PermutationGroups.lrx(4)
    assert graph.make_inverse_closed() == graph

    graph = PermutationGroups.lx(4).make_inverse_closed()
    assert graph.generators_permutations == [[1, 2, 3, 0], [1, 0, 2, 3], [3, 0, 1, 2]]
    assert graph.generator_names == ["L", "X", "L'"]
