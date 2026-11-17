import numpy as np
import pytest
import torch

from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import MatrixGenerator, CayleyGraphDef
from ..graphs_lib import PermutationGroups, MatrixGroups


def _state_to_str(state: torch.Tensor):
    return "".join(str(int(x)) for x in state)


def test_random_walks_single_walk():
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, y = graph.random_walks(width=1, length=5)
    assert x.shape == (5, 5)
    assert y.shape == (5,)
    assert _state_to_str(x[0]) == "01234"
    assert _state_to_str(x[1]) in ["12340", "40123", "10234"]
    assert np.array_equal(y.cpu().numpy(), [0, 1, 2, 3, 4])


def test_random_walks_matrix_group():
    graph = CayleyGraph(MatrixGroups.heisenberg())
    x, y = graph.random_walks(width=20, length=10)
    assert x.shape == (200, 3, 3)
    assert y.shape == (200,)
    assert np.array_equal(y.cpu().numpy(), [i for i in range(10) for _ in range(20)])


def test_random_walks_nbt_matrix_group():
    generators = [
        [[49, 1008], [1, 0]],
        [[0, 1], [1008, 49]],
        [[0, 583], [964, 49]],
        [[49, 426], [45, 0]],
    ]
    graph = CayleyGraph(
        CayleyGraphDef.for_matrix_group(
            generators=[MatrixGenerator.create(matrix, modulo=1009) for matrix in generators],
            generator_names=["a", "a'", "b", "b'"],
            central_state=[[1, 0], [0, 1]],
        ),
        device="cpu",
        random_seed=0,
    )
    width, length = 1000, 10
    x, y = graph.random_walks(width=width, length=length, mode="nbt")

    assert x.shape == (width * length, 2, 2)
    assert y.shape == (width * length,)
    # Verify that each walk is made of unique states.
    for i in range(width):
        walk = x[i::width].reshape(length, -1)
        assert torch.unique(walk, dim=0).shape[0] == length, f"Walk {i} revisits a state"


@pytest.mark.parametrize("width", [1, 2, 5])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 10])
@pytest.mark.parametrize("nbt_history_depth", [-1, 2])
def test_random_walks_nbt_exhaustion(width, length, nbt_history_depth):
    """Verify that NBT walks do not repeat states when no unseen candidates remain."""
    graph = CayleyGraph(
        CayleyGraphDef.for_matrix_group(
            generators=[MatrixGenerator.create([[value]], modulo=5) for value in (2, 3)],
            generator_names=["a", "b"],
            central_state=[[1]],
        ),
        device="cpu",
        random_seed=0,
    )
    x, y = graph.random_walks(width=width, length=length, mode="nbt", nbt_history_depth=nbt_history_depth)

    expected_length = min(length, 3)
    assert x.shape == (width * expected_length, 1, 1)
    expected_y = torch.tensor([0, 1, 2][:expected_length], dtype=y.dtype, device=y.device).repeat_interleave(width)
    assert torch.equal(y, expected_y), y
    for i in range(width):
        walk = x[i::width].reshape(expected_length, -1)
        assert torch.unique(walk, dim=0).shape[0] == expected_length, f"Walk {i} revisits a state"


@pytest.mark.parametrize("nbt_history_depth", [0, -2])
def test_random_walks_nbt_invalid_history_depth(nbt_history_depth):
    """Reject history depths that disable non-backtracking checks."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=0)

    with pytest.raises(ValueError, match=r"^nbt_history_depth must be positive\.$"):
        graph.random_walks(width=1, length=4, mode="nbt", nbt_history_depth=nbt_history_depth)


def test_random_walks_start_state():
    graph = CayleyGraph(PermutationGroups.lx(5))
    x, y = graph.random_walks(width=10, length=5, start_state=[1, 0, 0, 0, 0])
    assert x.shape == (50, 5)
    assert y.shape == (50,)
    for i in range(10):
        assert _state_to_str(x[i]) == "10000"
    for i in range(10, 20):
        assert _state_to_str(x[i]) in ["01000", "00001"]


def test_random_walks_bfs_small():
    graph = CayleyGraph(PermutationGroups.lrx(4))
    x, y = graph.random_walks(width=50, length=100, mode="bfs")
    assert x.shape == (24, 4)
    assert y.shape == (24,)
    assert np.array_equal(y.cpu().numpy(), [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6])


def test_random_walks_bfs():
    graph = CayleyGraph(PermutationGroups.lrx(20))
    x, y = graph.random_walks(width=100, length=50, mode="bfs")
    assert x.shape == (4485, 20)
    assert y.shape == (4485,)
    assert y[0] == 0
    assert y[-1] == 49


def test_random_walks_bfs_matrix_groups():
    graph = CayleyGraph(MatrixGroups.heisenberg())
    x, y = graph.random_walks(width=100, length=50, mode="bfs")
    assert x.shape == (4635, 3, 3)
    assert y.shape == (4635,)
