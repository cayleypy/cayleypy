import os

import pytest

from cayleypy import create_graph, PermutationGroups, CayleyGraph
from cayleypy import find_path
from cayleypy.algo.find_path import _precompute_bfs
from cayleypy.predictor import Predictor
from cayleypy.models.models_lib import PREDICTOR_MODELS

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def test_find_path_pancake8():
    graph = CayleyGraph(PermutationGroups.pancake(8))
    start_state = [4, 7, 3, 2, 0, 5, 1, 6]
    path = find_path(graph, start_state)
    assert path is not None
    graph.validate_path(start_state, path)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
@pytest.mark.parametrize("graph_name", ["lx-9", "lrx-9", "lrx-12", "lrx-15", "lrx-16", "cube_2/2/2_6gensQTM"])
def test_find_path(graph_name: str):
    graph = create_graph(name=graph_name)
    start_state = graph.random_walks(width=1, length=100)[0][-1]
    path = find_path(graph, start_state)
    assert path is not None
    graph.validate_path(start_state, path)


# =============================================================================
# _precompute_bfs caching (find_path.py:14-30)
# =============================================================================


def test_precompute_bfs_caches_result():
    """``_precompute_bfs`` caches the BFS result on the graph (find_path.py:14-31)."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    assert not hasattr(graph, "_bfs_result_cache")
    result1 = _precompute_bfs(graph)
    assert hasattr(graph, "_bfs_result_cache")
    # Second call with same params returns the cached object (no re-computation).
    result2 = _precompute_bfs(graph)
    assert result1 is result2


def test_precompute_bfs_different_params_different_cache():
    """A4 regression: different parameters produce different cached results."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    result_default = _precompute_bfs(graph)
    result_small = _precompute_bfs(graph, max_diameter=2)
    assert result_default is not result_small
    assert len(result_small.layer_sizes) <= 3  # layers 0, 1, 2


def test_precompute_bfs_returns_valid_result():
    """``_precompute_bfs`` returns a BfsResult with layer hashes."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    result = _precompute_bfs(graph)
    assert result is not None
    assert len(result.layer_sizes) > 0
    assert len(result.layers_hashes) == len(result.layer_sizes)


# =============================================================================
# find_path not-found cases
# =============================================================================


def test_find_path_unreachable_state_returns_none():
    """``find_path`` returns None when the start state is outside the BFS exploration radius.

    The state ``[7,6,5,4,2,1,0,3]`` is at distance 3 from central in pancake(8).
    With ``max_diameter=1`` (only layers 0-1 explored), this state is unreachable.
    """
    graph = CayleyGraph(PermutationGroups.pancake(8))
    path = find_path(graph, [7, 6, 5, 4, 2, 1, 0, 3], max_diameter=1)
    assert path is None


def test_find_path_non_inverse_closed_generators():
    """``find_path`` works for graphs with non-inverse-closed generators (find_path.py:76-81).

    For non-inverse-closed generators, ``find_path`` uses ``with_inverted_generators``
    and ``find_path_to`` instead of ``find_path_from``.
    """
    # LX graph has non-inverse-closed generators (L shift is not the inverse of itself).
    graph = CayleyGraph(PermutationGroups.lx(5))
    # Start from a state 1 step away.
    start_state = [1, 2, 3, 4, 0]
    path = find_path(graph, start_state, max_diameter=10)
    assert path is not None
    graph.validate_path(start_state, path)


def test_find_path_pretrained_with_explicit_beam_width(monkeypatch):
    """A3 regression: passing beam_width/max_steps to find_path must not raise TypeError.

    When a graph is in PREDICTOR_MODELS, find_path passes kwargs to beam_search.
    Previously kwargs.get(\"beam_width\") left the key in kwargs, causing duplicate
    keyword argument error on the beam_search call.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))
    # Monkeypatch PREDICTOR_MODELS so find_path takes the pretrained branch
    monkeypatch.setitem(PREDICTOR_MODELS, "lrx-5", object())
    # Monkeypatch Predictor.pretrained to return a Hamming predictor
    monkeypatch.setattr(Predictor, "pretrained", lambda g: Predictor(g, "hamming"))

    start_state = graph.random_walks(width=1, length=3)[0][-1]
    # This must not raise TypeError
    path = find_path(graph, start_state, beam_width=500, max_steps=10)
    # May or may not find a path (depends on randomness) — just check no TypeError
    assert path is None or isinstance(path, list)


def test_find_path_verbose_output(capsys):
    """``find_path`` with verbose graph prints pre-computation progress (find_path.py:18, 28)."""
    graph = CayleyGraph(PermutationGroups.lrx(5), verbose=1)
    capsys.readouterr()  # clear
    path = find_path(graph, [1, 0, 2, 3, 4], max_diameter=5)
    captured = capsys.readouterr()
    assert path is not None
    assert "Pre-computing" in captured.out or "Pre-computed" in captured.out
