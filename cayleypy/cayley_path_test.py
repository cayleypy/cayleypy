"""Tests for CayleyPath."""

import torch

from cayleypy import CayleyGraph, PermutationGroups
from cayleypy.cayley_path import CayleyPath


def test_cayley_path_all_states():
    """CayleyPath.all_states returns correct states for a multi-step path."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [0, 1, 2, 3, 4]
    edges = [0, 2]  # L then X
    path = CayleyPath(start_state=graph.encode_states(start), edges=edges, graph=graph.definition)
    states = path.all_states
    assert len(states) == 3
    # First element is the raw start_state (2D from encode_states); subsequent are 1D
    assert torch.equal(states[0], graph.encode_states(start))
    assert torch.equal(states[-1], graph.apply_path(start, edges)[0])


def test_cayley_path_end_state():
    """CayleyPath.end_state equals the last element of all_states."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [0, 1, 2, 3, 4]
    edges = [0, 2]
    path = CayleyPath(start_state=graph.encode_states(start), edges=edges, graph=graph.definition)
    assert torch.equal(path.end_state, path.all_states[-1])


def test_cayley_path_empty():
    """CayleyPath with no edges: all_states == [start], end_state == start."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [0, 1, 2, 3, 4]
    path = CayleyPath(start_state=graph.encode_states(start), edges=[], graph=graph.definition)
    assert len(path.all_states) == 1
    assert torch.equal(path.all_states[0], graph.encode_states(start))
    assert torch.equal(path.end_state, path.all_states[0])


def test_cayley_path_repr():
    """CayleyPath.__repr__ returns a non-empty string."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [0, 1, 2, 3, 4]
    edges = [0]
    path = CayleyPath(start_state=graph.encode_states(start), edges=edges, graph=graph.definition)
    r = repr(path)
    assert isinstance(r, str)
    assert len(r) > 0
