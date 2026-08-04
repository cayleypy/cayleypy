import os

import pytest
import torch

from .algo.bfs_result import BfsResult
from .cayley_graph import CayleyGraph
from .graphs_lib import PermutationGroups
from .lower_bound import BfsLowerBound, LowerBound
from .puzzles import Puzzles

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def exact_distances(bfs_result: BfsResult) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns all states of a completed BFS and their exact distances from the central state."""
    assert bfs_result.bfs_completed
    distances = [torch.full((n,), i, dtype=torch.int64) for i, n in enumerate(bfs_result.layer_sizes)]
    return bfs_result.all_states, torch.cat(distances)


def test_lower_bound_is_exact_when_bfs_is_complete():
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    bfs_result = graph.bfs(max_layer_size_to_store=None, return_all_hashes=True)
    states, distances = exact_distances(bfs_result)

    lower_bound = BfsLowerBound(graph, bfs_result)

    assert lower_bound.radius == bfs_result.diameter()
    assert torch.equal(lower_bound.lb(states), distances)


def test_lower_bound_is_exact_inside_ball_and_admissible_outside():
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    states, distances = exact_distances(graph.bfs(max_layer_size_to_store=None, return_all_hashes=True))
    radius = 4

    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=radius, return_all_hashes=True))

    bounds = lower_bound.lb(states)
    assert lower_bound.radius == radius
    # Admissibility: the bound never exceeds the true distance.
    assert bool(torch.all(bounds <= distances))
    inside = distances <= radius
    assert bool(torch.all(bounds[inside] == distances[inside]))
    # Everything outside the ball is at distance at least radius+1, and that is what is returned for it.
    assert bool(torch.all(bounds[~inside] == radius + 1))
    assert int((~inside).sum()) > 0


def test_lower_bound_for_single_state():
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=2, return_all_hashes=True))

    assert lower_bound.lb(graph.central_state).tolist() == [0]
    assert lower_bound.lb([1, 0, 2, 3, 4, 5]).tolist() == [1]
    assert lower_bound.lb([5, 4, 3, 2, 1, 0]).tolist() == [3]


def test_lower_bound_for_puzzle_with_colors():
    """Test on a graph where states are colors of stickers, not permutations."""
    graph = CayleyGraph(Puzzles.rubik_cube(2, metric="QTM"), device="cpu")
    radius = 3
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=radius, return_all_hashes=True))

    assert lower_bound.lb(graph.central_state).tolist() == [0]
    # f0, r0, d0 - quarter turns of three different faces, which do not undo each other.
    path = [graph.definition.generator_names.index(name) for name in ["f0", "r0", "d0"]]
    for n_moves in range(1, radius + 1):
        state = graph.apply_path(graph.central_state, path[:n_moves])
        assert lower_bound.lb(state).tolist() == [n_moves]
    far_state = graph.apply_path(graph.central_state, path * 3)
    assert lower_bound.lb(far_state).tolist() == [radius + 1]


def test_lower_bound_batch_matches_state_by_state():
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=3, return_all_hashes=True))
    states = graph.random_walks(width=20, length=6)[0]

    bounds = lower_bound.lb(states)

    assert bounds.shape == (states.shape[0],)
    for i in range(states.shape[0]):
        assert lower_bound.lb(states[i]).tolist() == [int(bounds[i])]


def test_bfs_lower_bound_implements_lower_bound_protocol():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=1, return_all_hashes=True))

    assert isinstance(lower_bound, LowerBound)
    # Any object with the `lb` method can be used as a lower bound.
    assert isinstance(_ZeroLowerBound(), LowerBound)
    assert not isinstance(graph, LowerBound)


class _ZeroLowerBound:
    """Trivial (and useless, but admissible) lower bound."""

    def lb(self, states: torch.Tensor) -> torch.Tensor:
        return torch.zeros((states.shape[0],), dtype=torch.int64)


def test_lower_bound_requires_hashes():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    with pytest.raises(ValueError, match="return_all_hashes"):
        BfsLowerBound(graph, graph.bfs(max_diameter=2))


def test_lower_bound_requires_inverse_closed_generators():
    graph_def = PermutationGroups.lx(5)
    assert not graph_def.generators_inverse_closed
    graph = CayleyGraph(graph_def, device="cpu")

    with pytest.raises(ValueError, match="inverse-closed"):
        BfsLowerBound(graph, graph.bfs(max_diameter=2, return_all_hashes=True))


def test_lower_bound_rejects_bfs_for_other_graph():
    graph1 = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    graph2 = CayleyGraph(PermutationGroups.lrx(5, k=2), device="cpu")

    with pytest.raises(ValueError, match="different graph"):
        BfsLowerBound(graph1, graph2.bfs(max_diameter=2, return_all_hashes=True))


def test_lower_bound_rejects_bfs_from_other_state():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    bfs_result = graph.bfs(start_states=[[1, 0, 2, 3, 4]], max_diameter=2, return_all_hashes=True)

    with pytest.raises(ValueError, match="started from the central state"):
        BfsLowerBound(graph, bfs_result)


def test_lower_bound_rejects_bfs_from_other_graph_object():
    """Test that BFS run by another graph object is rejected, because that object hashes states differently."""
    # State of this graph does not fit in one int64, so hashes depend on the random seed of the graph object.
    graph_def = PermutationGroups.lrx(20)
    graph1 = CayleyGraph(graph_def, device="cpu")
    graph2 = CayleyGraph(graph_def, device="cpu")
    assert graph1.hasher.seed != graph2.hasher.seed

    with pytest.raises(ValueError, match="same CayleyGraph object"):
        BfsLowerBound(graph1, graph2.bfs(max_diameter=2, return_all_hashes=True))


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_lower_bound_on_cube222_is_admissible():
    """Test admissibility on the 2x2x2 cube, comparing a bound from a small ball with exact distances."""
    graph = CayleyGraph(Puzzles.rubik_cube(2, metric="QTM"), device="cpu")
    # Exact distances for all states reachable in 8 moves (there are almost 4 million of them).
    exact = BfsLowerBound(graph, graph.bfs(max_diameter=8, return_all_hashes=True, max_layer_size_to_store=1))
    radius = 5
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=radius, return_all_hashes=True))
    # Random walks of length 8 stay inside the ball for which distances are known exactly.
    states = graph.random_walks(width=1000, length=9)[0]

    bounds = lower_bound.lb(states)
    distances = exact.lb(states)

    assert bool(torch.all(bounds <= distances))
    inside = distances <= radius
    assert bool(torch.all(bounds[inside] == distances[inside]))
    assert bool(torch.all(bounds[~inside] == radius + 1))
    assert int((~inside).sum()) > 0
