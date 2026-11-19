import numpy as np
import pytest
import torch

from cayleypy import CayleyGraphDef, MatrixGenerator, PermutationGroups


def test_inverse_permutations() -> None:
    graph_def = CayleyGraphDef.create([[1, 0, 2, 3], [1, 2, 3, 0], [0, 2, 3, 1]])
    inv = graph_def.with_inverted_generators()
    assert inv.is_permutation_group()
    assert inv.generators_permutations == [[1, 0, 2, 3], [3, 0, 1, 2], [0, 3, 1, 2]]


@pytest.mark.parametrize("modulo", [0, 10, 17])
def test_inverse_matrices(modulo: int) -> None:
    x = MatrixGenerator.create([[1, 1, 0], [0, 1, 0], [0, 0, 1]], modulo=modulo)
    x_inv = MatrixGenerator.create([[1, -1, 0], [0, 1, 0], [0, 0, 1]], modulo=modulo)
    graph_def = CayleyGraphDef.for_matrix_group(generators=[x])
    inv = graph_def.with_inverted_generators()
    assert inv.is_matrix_group()
    assert len(inv.generators_matrices) == 1
    assert inv.generators_matrices[0] == x_inv


def test_make_inverse_closed() -> None:
    graph = PermutationGroups.lrx(4)
    assert graph.make_inverse_closed() == graph

    graph = PermutationGroups.lx(4).make_inverse_closed()
    assert graph.generators_permutations == [[1, 2, 3, 0], [1, 0, 2, 3], [3, 0, 1, 2]]
    assert graph.generator_names == ["L", "X", "L'"]


@pytest.mark.parametrize("modulo", [5, 2**31 - 1, 2**31])
def test_modular_matrix_multiplication_overflow(modulo: int) -> None:
    """Compare NumPy and batched Torch multiplication with exact arithmetic."""
    generator = MatrixGenerator.create([[-3, -2, -2], [-2, -3, -2], [-2, -2, -3]], modulo=modulo)
    state = np.array([[2**63 - 1, -(2**63)], [-1, 1], [modulo - 1, 0]], dtype=np.int64)
    states = np.stack([state, state[::-1]])
    expected = np.asarray((generator.matrix.astype(object) @ states.astype(object)) % modulo, dtype=np.int64)

    assert generator.is_inverse_to(generator.inv)
    assert np.array_equal(generator.apply(state), expected[0])
    assert torch.equal(generator.apply_batch_torch(torch.from_numpy(states)), torch.from_numpy(expected))
