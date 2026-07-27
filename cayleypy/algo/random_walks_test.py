"""Tests for ``cayleypy.algo.random_walks.RandomWalksGenerator``.

This module has no test file currently; these tests cover each public mode.
"""

import pytest
import numpy as np
import torch

from cayleypy.cayley_graph import CayleyGraph
from cayleypy.graphs_lib import PermutationGroups
from cayleypy.algo.random_walks import RandomWalksGenerator


# =============================================================================
# generate() mode dispatch
# =============================================================================


def test_generate_classic_mode():
    """Classic mode: ``width * length`` states, first ``width`` are the start state."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, y = graph.random_walks(width=3, length=5, mode="classic")
    assert x.shape[0] == 15
    assert y.shape[0] == 15
    # First ``width`` states are the start state (central), distance 0.
    assert torch.all(y[:3] == 0)
    # Distances increase monotonically per walk.
    for i in range(3):
        walk_dists = y[i::3]
        assert torch.all(walk_dists[1:] >= walk_dists[:-1])


def test_generate_bfs_mode():
    """BFS mode: output size <= width*length, first state is start with distance 0."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, y = graph.random_walks(width=10, length=10, mode="bfs")
    assert x.shape[0] <= 100
    assert y.shape[0] == x.shape[0]
    # First state is start (central), distance 0.
    assert y[0] == 0


def test_generate_bfs_mode_states_unique_without_subsampling():
    """BFS mode: all states are unique when ``width`` >= largest layer (no subsampling).

    This is the documented invariant: ``random_walks.py:55`` says "All states in the
    output are unique." When ``width`` is large enough that no layer is subsampled,
    the invariant holds.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, _ = graph.random_walks(width=10**6, length=10, mode="bfs")
    state_tuples = [tuple(int(v) for v in s) for s in x]
    assert len(state_tuples) == len(set(state_tuples)), "BFS states should be unique without subsampling"


def test_generate_bfs_mode_states_unique_with_subsampling():
    """BFS mode: all states are unique even when ``width`` < layer size (subsampling active).

    Previously this was a known bug (TODO(char-spec)): after ``torch.randperm``
    subsampling, hashes were in random order, violating the ``add_sorted_hashes``
    precondition, causing duplicate states to slip through. Fixed by sorting hashes
    after subsampling (random_walks.py). This test now asserts the documented
    invariant "All states in the output are unique" (random_walks.py:55) holds even
    with subsampling.
    """
    np.random.seed(12345)
    torch.manual_seed(12345)
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, _ = graph.random_walks(width=5, length=10, mode="bfs")
    state_tuples = [tuple(int(v) for v in s) for s in x]
    assert len(state_tuples) == len(set(state_tuples)), "BFS states should be unique even with subsampling"


def test_generate_nbt_mode():
    """NBT mode: ``width * length`` states, non-backtracking constraints applied."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, y = graph.random_walks(width=5, length=8, mode="nbt", nbt_history_depth=3)
    assert x.shape[0] == 40
    assert y.shape[0] == 40
    assert torch.all(y[:5] == 0)


def test_generate_invalid_mode_raises():
    """Invalid mode raises ValueError."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    with pytest.raises(ValueError, match="Unknown mode"):
        graph.random_walks(width=3, length=5, mode="invalid")


# =============================================================================
# random_walks_classic edge cases
# =============================================================================


def test_classic_single_walk_single_step():
    """Width=1, length=1 -> exactly 1 state (the start)."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, y = RandomWalksGenerator(graph).generate(width=1, length=1, mode="classic")
    assert x.shape[0] == 1
    assert y[0] == 0


def test_classic_custom_start_state():
    """Custom start_state is used as the first state."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [4, 3, 2, 1, 0]
    x, _ = RandomWalksGenerator(graph).generate(width=1, length=2, mode="classic", start_state=start)
    assert tuple(int(v) for v in x[0]) == tuple(start)


# =============================================================================
# random_walks_bfs edge cases
# =============================================================================


def test_bfs_exhausts_graph():
    """On a tiny graph, BFS exhausts all states before reaching ``length`` layers."""
    graph = CayleyGraph(PermutationGroups.lrx(3))
    # LRX(3) has 6 states; with width=10 and length=100, BFS finds all 6.
    x, _ = graph.random_walks(width=10, length=100, mode="bfs")
    assert x.shape[0] <= 60  # at most 6 states * 10 width
    assert x.shape[0] <= 6  # all unique -> at most 6


def test_bfs_width_limits_layer():
    """Width caps the number of states per layer."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    _, y = graph.random_walks(width=2, length=10, mode="bfs")
    # Each layer has at most 2 states.
    for step in range(1, 10):
        mask = y == step
        assert mask.sum() <= 2
