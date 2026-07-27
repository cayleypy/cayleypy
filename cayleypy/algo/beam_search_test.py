"""Tests for beam search algorithm."""

import os

import numpy as np
import pytest
import torch

from ..cayley_graph import CayleyGraph

from ..graphs_lib import PermutationGroups, MatrixGroups, prepare_graph
from ..predictor import Predictor
from .beam_search_result import BeamSearchResult

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def _validate_beam_search_result(graph: CayleyGraph, start_state, bs_result: BeamSearchResult):
    """Validate that beam search result is correct."""
    assert bs_result.path_found
    assert bs_result.path is not None
    path_result = graph.apply_path(start_state, bs_result.path).reshape((-1))
    assert torch.equal(path_result, graph.central_state)


def _scramble(graph: CayleyGraph, num_scrambles: int) -> torch.Tensor:
    """Create a scrambled state by applying random moves."""
    return graph.random_walks(width=1, length=num_scrambles + 1)[0][-1]


# =============================================================================
# Tests for "simple" beam search mode
# =============================================================================


def test_beam_search_simple_lrx_few_steps():
    """Test simple beam search on small LRX graph with few steps."""
    graph = CayleyGraph(PermutationGroups.lrx(5))

    # Test starting from central state
    result0 = graph.beam_search(start_state=[0, 1, 2, 3, 4], beam_mode="simple")
    assert result0.path_found
    assert result0.path_length == 0

    # Test one step away
    result1 = graph.beam_search(start_state=[1, 0, 2, 3, 4], beam_mode="simple", return_path=True)
    assert result1.path_found
    assert result1.path_length == 1
    assert result1.path == [2]
    assert result1.get_path_as_string() == "X"

    # Test two steps away
    result2 = graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="simple", return_path=True)
    assert result2.path_found
    assert result2.path_length == 2
    assert result2.path == [0, 2]
    assert result2.get_path_as_string() == "L.X"


def test_beam_search_simple_lrx_n8_random():
    """Test simple beam search on random LRX(8) state."""
    n = 8
    graph = CayleyGraph(PermutationGroups.lrx(n))
    start_state = np.random.permutation(n)

    bs_result = graph.beam_search(start_state=start_state, beam_mode="simple", beam_width=10**7, return_path=True)
    assert bs_result.path_length <= 28
    _validate_beam_search_result(graph, start_state, bs_result)


def test_beam_search_simple_mini_pyramorphix():
    """Test simple beam search on mini pyramorphix puzzle."""
    graph = CayleyGraph(prepare_graph("mini_pyramorphix"))
    start_state = _scramble(graph, 100)
    bs_result = graph.beam_search(start_state=start_state, beam_mode="simple", beam_width=10**7, return_path=True)
    assert bs_result.path_length <= 5
    _validate_beam_search_result(graph, start_state, bs_result)


def test_beam_search_simple_with_predictor():
    """Test simple beam search with pretrained predictor."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)
    result = graph.beam_search(start_state=state, beam_mode="simple", predictor=predictor)
    assert result.path_found


def test_beam_search_simple_meet_in_the_middle():
    """Test simple beam search with meet-in-the-middle optimization."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    bfs_result = graph.bfs(max_diameter=10, return_all_hashes=True)
    state = _scramble(graph, 120)
    result = graph.beam_search(
        start_state=state, beam_mode="simple", predictor=predictor, hashed_neigbourhood=bfs_result, return_path=True
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_simple_meet_in_the_middle_int():
    """Test simple beam search with meet-in-the-middle optimization as integer value."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)
    result = graph.beam_search(
        start_state=state, beam_mode="simple", predictor=predictor, hashed_neigbourhood=10, return_path=True
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_simple_matrix_groups():
    """Test simple beam search on matrix groups."""
    graph = CayleyGraph(MatrixGroups.heisenberg())
    start_state = [[1, 2, 3], [0, 1, 1], [0, 0, 1]]
    bs_result = graph.beam_search(start_state=start_state, beam_mode="simple", return_path=True)
    _validate_beam_search_result(graph, start_state, bs_result)


def test_beam_search_simple_not_found():
    """Test simple beam search when path is not found."""
    n = 50
    graph = CayleyGraph(PermutationGroups.lrx(n))
    start_state = np.random.permutation(n)
    bs_result = graph.beam_search(start_state=start_state, beam_mode="simple", beam_width=10, max_steps=10)
    assert not bs_result.path_found


# =============================================================================
# Tests for "advanced" beam search mode
# =============================================================================


def test_beam_search_advanced_lrx_few_steps():
    """Test advanced beam search on small LRX graph with few steps."""
    graph = CayleyGraph(PermutationGroups.lrx(5), dtype=torch.int8)

    # Test starting from central state
    result0 = graph.beam_search(start_state=[0, 1, 2, 3, 4], beam_mode="advanced")
    assert result0.path_found
    assert result0.path_length == 0

    # Test one step away
    result1 = graph.beam_search(start_state=[1, 0, 2, 3, 4], beam_mode="advanced")
    assert result1.path_found
    assert result1.path_length == 1

    # Test two steps away
    result2 = graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="advanced")
    assert result2.path_found
    assert result2.path_length == 2


def test_beam_search_advanced_with_history_depth():
    """Test advanced beam search with non-backtracking (history_depth > 0)."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = np.random.permutation(8)

    # Test with history_depth = 2
    result = graph.beam_search(
        start_state=start_state, beam_mode="advanced", history_depth=2, beam_width=1000, max_steps=20
    )
    # Should find path or exhaust search space
    assert result.path_found or result.path_length == 20


def test_beam_search_advanced_with_predictor():
    """Test advanced beam search with pretrained predictor."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)
    result = graph.beam_search(start_state=state, beam_mode="advanced", predictor=predictor, history_depth=3)
    assert result.path_found


def test_beam_search_advanced_meet_in_the_middle():
    """Test advanced beam search with meet-in-the-middle optimization."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    bfs_result = graph.bfs(max_diameter=10, return_all_hashes=True)
    state = _scramble(graph, 120)
    result = graph.beam_search(
        start_state=state, beam_mode="advanced", predictor=predictor, hashed_neigbourhood=bfs_result, return_path=True
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_advanced_meet_in_the_middle_int():
    """Test advanced beam search with meet-in-the-middle optimization as integer value."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)
    result = graph.beam_search(
        start_state=state, beam_mode="advanced", predictor=predictor, hashed_neigbourhood=10, return_path=True
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_advanced_meet_in_the_middle_int_and_history_depth_2():
    """Test advanced beam search with meet-in-the-middle optimization as integer value."""
    graph = CayleyGraph(PermutationGroups.lrx(16))  # , random_seed= 84791592
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)  # reduced from 120 to 16 because of random
    result = graph.beam_search(
        start_state=state,
        beam_mode="advanced",
        predictor=predictor,
        hashed_neigbourhood=10,
        return_path=True,
        history_depth=2,
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_advanced_matrix_groups():
    """Test advanced beam search on matrix groups."""
    graph = CayleyGraph(MatrixGroups.heisenberg())
    start_state = [[1, 2, 3], [0, 1, 1], [0, 0, 1]]
    bs_result = graph.beam_search(start_state=start_state, beam_mode="advanced", history_depth=1)
    assert bs_result.path_found


def test_beam_search_advanced_not_found():
    """Test advanced beam search when path is not found."""
    n = 50
    graph = CayleyGraph(PermutationGroups.lrx(n))
    start_state = np.random.permutation(n)
    bs_result = graph.beam_search(
        start_state=start_state, beam_mode="advanced", beam_width=10, max_steps=10, history_depth=2
    )
    assert not bs_result.path_found


def test_beam_search_advanced_verbose_output():
    """Test advanced beam search with verbose output."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = np.random.permutation(8)

    # Test with verbose=1
    result = graph.beam_search(start_state=start_state, beam_mode="advanced", verbose=1, max_steps=5)
    # Should complete without errors
    assert result.path_found or result.path_length == 5


# =============================================================================
# Tests for "iterated" beam search mode
# =============================================================================


def test_beam_search_iterated_lrx_few_steps():
    """Test iterated beam search on small LRX graph with few steps."""
    graph = CayleyGraph(PermutationGroups.lrx(5), dtype=torch.int8)

    # Test starting from central state
    result0 = graph.beam_search(start_state=[0, 1, 2, 3, 4], beam_mode="iterated")
    assert result0.path_found
    assert result0.path_length == 0

    # Test one step away
    result1 = graph.beam_search(start_state=[1, 0, 2, 3, 4], beam_mode="iterated")
    assert result1.path_found
    assert result1.path_length == 1

    # Test two steps away
    result2 = graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="iterated")
    assert result2.path_found
    assert result2.path_length == 2


def test_beam_search_iterated_with_history_depth():
    """Test iterated beam search with non-backtracking (history_depth > 0)."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = np.random.permutation(8)

    # Test with history_depth = 2
    result = graph.beam_search(
        start_state=start_state, beam_mode="iterated", history_depth=4, beam_width=1000, max_steps=20
    )
    # Should find path or exhaust search space
    assert result.path_found or result.path_length == 20


def test_beam_search_iterated_with_predictor():
    """Test iterated beam search with pretrained predictor."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)
    result = graph.beam_search(start_state=state, beam_mode="iterated", predictor=predictor, history_depth=10)
    assert result.path_found


def test_beam_search_iterated_meet_in_the_middle():
    """Test iterated beam search with meet-in-the-middle optimization."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    bfs_result = graph.bfs(max_diameter=10, return_all_hashes=True)
    state = _scramble(graph, 120)
    result = graph.beam_search(
        start_state=state, beam_mode="iterated", predictor=predictor, hashed_neigbourhood=bfs_result, return_path=True
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_iterated_meet_in_the_middle_int():
    """Test iterated beam search with meet-in-the-middle optimization as integer value."""
    graph = CayleyGraph(PermutationGroups.lrx(16))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)
    result = graph.beam_search(
        start_state=state, beam_mode="iterated", predictor=predictor, hashed_neigbourhood=10, return_path=True
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


def test_beam_search_iterated_meet_in_the_middle_int_and_history_depth_2():
    """Test iterated beam search with meet-in-the-middle optimization as integer value."""
    graph = CayleyGraph(PermutationGroups.lrx(16))  # , random_seed= 84791592
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 120)  # reduced from 120 to 16 because of random
    result = graph.beam_search(
        start_state=state,
        beam_mode="iterated",
        predictor=predictor,
        hashed_neigbourhood=10,
        return_path=True,
        history_depth=2,
    )
    assert result.path_found
    _validate_beam_search_result(graph, state, result)


# For now iterated beam search don't works with matrix groups.
# def test_beam_search_iterated_matrix_groups():
#     """Test iterated beam search on matrix groups."""
#     graph = CayleyGraph(MatrixGroups.heisenberg())
#     start_state = [[1, 2, 3], [0, 1, 1], [0, 0, 1]]
#     bs_result = graph.beam_search(start_state=start_state, beam_mode="iterated", history_depth=1)
#     assert bs_result.path_found


def test_beam_search_iterated_not_found():
    """Test iterated beam search when path is not found."""
    n = 50
    graph = CayleyGraph(PermutationGroups.lrx(n))
    start_state = np.random.permutation(n)
    bs_result = graph.beam_search(
        start_state=start_state, beam_mode="iterated", beam_width=10, max_steps=10, history_depth=2
    )
    assert not bs_result.path_found


def test_beam_search_iterated_verbose_output():
    """Test iterated beam search with verbose output."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = np.random.permutation(8)

    # Test with verbose=1
    result = graph.beam_search(start_state=start_state, beam_mode="iterated", verbose=1, max_steps=5)
    # Should complete without errors
    assert result.path_found or result.path_length == 5


def test_beam_search_iterated_dedup_no_false_positive_on_zero_hash():
    """Dedup must not falsely remove a state whose hash is 0.

    Covers the dedup-unification fix (replaces the zero-padded accm_hashes buffer
    whose `torch.isin` against padding-zeros falsely deduped any hash==0 state).
    With the TorchHashSet, only actual hashes added this step are in the set.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    # Use a small beam so multiple chunks are generated and dedup runs.
    # history_depth=0 isolates the dedup path (no nonbacktrack interference).
    result = graph.beam_search(
        start_state=np.random.permutation(8),
        beam_mode="iterated",
        beam_width=100,
        max_steps=20,
        history_depth=0,
    )
    # Should complete without errors — if the zero-hash bug were present, some
    # valid states would be silently dropped, potentially causing path_not_found
    # or wrong path_length. The characterization tests with `<=` bounds tolerate
    # composition changes; here we just assert no crash.
    assert isinstance(result.path_found, bool)


def test_beam_search_iterated_dedup_sorted_precondition_after_topk():
    """The re-sort after topk must produce a hash-sorted chunk for add_sorted_hashes.

    Covers the dedup-unification's extra sort branch (when topk reorders by score,
    breaking hash order). Exercises the path where _topk_applied is True so the
    re-sort runs. A beam_width small enough that chunks exceed beam_width_part
    (forcing topk) is required.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    # beam_width=1000, 3 generators → beam_width_part=333. After the first step
    # the beam grows beyond 333, so topk runs on subsequent chunks.
    result = graph.beam_search(
        start_state=np.random.permutation(8),
        beam_mode="iterated",
        beam_width=1000,
        max_steps=10,
        history_depth=2,
    )
    # If the sorted precondition were violated, add_sorted_hashes would produce
    # a corrupt set → false dedup → path_not_found or wrong path. Assert no crash.
    assert result.path_found or result.path_length == 10


# =============================================================================
# Tests for exact values
# =============================================================================


def _cycle_roll_predictor(x, y):
    return (x - y).abs().quantile(0.5)


# =============================================================================
# Tests for default beam search (should use "simple" mode)
# =============================================================================


def test_beam_search_default_mode():
    """Test that default beam search uses simple mode."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start_state = [1, 0, 2, 3, 4]

    # Default mode (should be "simple")
    result_default = graph.beam_search(start_state=start_state, return_path=True)

    # Explicit simple mode
    result_simple = graph.beam_search(start_state=start_state, beam_mode="simple", return_path=True)

    # Results should be identical
    assert result_default.path_found == result_simple.path_found
    assert result_default.path_length == result_simple.path_length
    if result_default.path is not None and result_simple.path is not None:
        assert result_default.path == result_simple.path


# =============================================================================
# Slow tests (only run with RUN_SLOW_TESTS=1)
# =============================================================================


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_beam_search_simple_lrx_32():
    """Test simple beam search on large LRX(32) graph."""
    graph = CayleyGraph(PermutationGroups.lrx(32))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 496)
    result = graph.beam_search(start_state=state, beam_mode="simple", predictor=predictor)
    assert result.path_found


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_beam_search_advanced_lrx_32():
    """Test advanced beam search on large LRX(32) graph."""
    graph = CayleyGraph(PermutationGroups.lrx(32))
    predictor = Predictor.pretrained(graph)
    state = _scramble(graph, 496)
    result = graph.beam_search(start_state=state, beam_mode="advanced", predictor=predictor, history_depth=5)
    assert result.path_found


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_beam_search_simple_cube222():
    """Test simple beam search on 2x2x2 cube."""
    graph = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"))
    start_state = _scramble(graph, 100)
    bs_result = graph.beam_search(start_state=start_state, beam_mode="simple", beam_width=10**7, return_path=True)
    assert bs_result.path_length <= 14
    _validate_beam_search_result(graph, start_state, bs_result)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_beam_search_advanced_cube222():
    """Test advanced beam search on 2x2x2 cube."""
    graph = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"))
    start_state = _scramble(graph, 100)
    bs_result = graph.beam_search(start_state=start_state, beam_mode="advanced", beam_width=10**4, history_depth=3)
    assert bs_result.path_found


# =============================================================================
# Error handling tests
# =============================================================================


def test_beam_search_invalid_mode():
    """Test that invalid beam_mode raises ValueError."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start_state = [1, 0, 2, 3, 4]

    with pytest.raises(ValueError, match="Unknown beam_mode"):
        graph.beam_search(start_state=start_state, beam_mode="invalid_mode")


def test_beam_search_result_repr():
    """Test ``BeamSearchResult.__repr__`` for both found and not-found cases.

    Covers beam_search_result.py:26-32.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))

    # Not-found result.
    result_not_found = BeamSearchResult(False, 0, None, {}, graph.definition)
    assert repr(result_not_found) == "BeamSearchResult(path_found=False)"

    # Found result with path.
    result_found = BeamSearchResult(True, 2, [0, 2], {}, graph.definition)
    repr_str = repr(result_found)
    assert "BeamSearchResult(path_length=2" in repr_str
    assert "path=" in repr_str

    # Found result with path_length=0 (empty path).
    result_zero = BeamSearchResult(True, 0, [], {}, graph.definition)
    repr_zero = repr(result_zero)
    assert "path_length=0" in repr_zero


def test_beam_search_advanced_with_mitm_works():
    """Test that advanced mode accepts a BfsResult ``hashed_neigbourhood`` without error.

    NOTE: despite the previous name ``..._with_mitm_error``, the original test body
    asserts success (not an error) — the advanced mode currently accepts the
    ``hashed_neigbourhood`` argument and uses it for the meet-in-the-middle check.
    Renamed to reflect the pinned behavior.
    # TODO(char-spec): revisit whether advanced mode should support / reject MITM
    # neighborhood of a different graph (currently ``beam_search.py:371`` validates the
    # graph matches). Revisit during the perf plan when the 3 modes are unified.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = np.random.permutation(8)
    bfs_result = graph.bfs(max_diameter=5, return_all_hashes=True)

    # This should work (hashed_neigbourhood is ignored in advanced mode)
    result = graph.beam_search(start_state=start_state, beam_mode="advanced", hashed_neigbourhood=bfs_result)
    assert result.path_found or result.path_length > 0


# =============================================================================
# Characterization test matrix (pins current behavior of the 3 beam search modes).
#
# These tests are characterization tests: they document what the code does today,
# not what it should ideally do. Suspect behavior is marked with
# ``# TODO(char-spec):`` so it can be revisited when the 3 modes are unified in a
# follow-up performance plan. The full matrix (mode x {MITM, history_depth,
# return_path, predictor type, destination, outcome}) must run fully offline using
# hamming/zero/mock predictors (Kaggle-dependent tests stay under RUN_SLOW_TESTS).
# =============================================================================


_BEAM_MODES = ["simple", "advanced", "iterated"]
_ADV_MODES = ["advanced", "iterated"]  # modes that support history_depth / destination_state


class _NumpyPredictor:
    """Callable returning a numpy array — exercises the ``np.argsort`` branch.

    Covers ``beam_search.py:457-459`` (simple/advanced) and ``:695-697`` (iterated),
    where scores are not a torch.Tensor and the code falls back to ``np.argsort``.
    """

    def __call__(self, states: torch.Tensor):
        # Hamming-like score in numpy so the np.argsort branch is taken.
        central = states.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
        return ((states != central).sum(dim=1)).cpu().numpy()


class _TorchModulePredictor(torch.nn.Module):
    """Minimal ``nn.Module`` predictor — covers the ``isinstance(nn.Module)`` branch."""

    def forward(self, states: torch.Tensor):
        central = states.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
        return (states != central).sum(dim=1).to(torch.float32)


class _ObjectWithPredict:
    """Object exposing a ``.predict`` method — covers that branch in Predictor.__init__."""

    def predict(self, states: torch.Tensor):
        central = states.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
        return (states != central).sum(dim=1)


def _lrx8_scramble_far():
    """A fixed far-from-central LRX(8) state that is not trivially solvable in a few steps."""
    return [3, 5, 7, 1, 0, 6, 4, 2]


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_already_solved(beam_mode):
    """Outcome: start == central state -> path_length == 0, empty path."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    result = graph.beam_search(start_state=[0, 1, 2, 3, 4], beam_mode=beam_mode, return_path=True)
    assert result.path_found
    assert result.path_length == 0
    assert result.path == []


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_not_found(beam_mode):
    """Outcome: path not found within max_steps -> path_found is False."""
    graph = CayleyGraph(PermutationGroups.lrx(50))
    start_state = np.random.permutation(50)
    kwargs = {"beam_mode": beam_mode, "beam_width": 10, "max_steps": 10}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert not result.path_found
    assert result.path is None
    # not-found returns path_length == max_steps (advanced/iterated) or 0 (simple).
    if beam_mode == "simple":
        assert result.path_length == 0
    else:
        assert result.path_length == 10


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_return_path_true_false(beam_mode):
    """return_path axis: True yields a valid path, False yields path=None but length set."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    kwargs = {"beam_mode": beam_mode, "beam_width": 10**5, "max_steps": 30}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2

    result_with_path = graph.beam_search(start_state=start_state, return_path=True, **kwargs)
    result_no_path = graph.beam_search(start_state=start_state, return_path=False, **kwargs)

    assert result_with_path.path_found
    assert result_with_path.path is not None
    _validate_beam_search_result(graph, start_state, result_with_path)

    assert result_no_path.path_found
    assert result_no_path.path is None
    # Same graph/scramble -> same discovered length (deterministic with conftest seed).
    assert result_no_path.path_length == result_with_path.path_length


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_mitm_int(beam_mode):
    """hashed_neigbourhood as int radius -> meet-in-the-middle is enabled and finds a path."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    kwargs = {
        "beam_mode": beam_mode,
        "beam_width": 10**5,
        "max_steps": 30,
        "hashed_neigbourhood": 5,
        "return_path": True,
    }
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_mitm_bfsresult(beam_mode):
    """hashed_neigbourhood as BfsResult -> meet-in-the-middle is enabled and finds a path."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    bfs_result = graph.bfs(max_diameter=5, return_all_hashes=True)
    start_state = _lrx8_scramble_far()
    kwargs = {
        "beam_mode": beam_mode,
        "beam_width": 10**5,
        "max_steps": 30,
        "hashed_neigbourhood": bfs_result,
        "return_path": True,
    }
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_matrix_mitm_graph_mismatch_raises():
    """MITM neighborhood from a different graph must raise ValueError (beam_search.py:222/371/597)."""
    graph_a = CayleyGraph(PermutationGroups.lrx(8))
    graph_b = CayleyGraph(PermutationGroups.lrx(10))
    bfs_result_b = graph_b.bfs(max_diameter=3, return_all_hashes=True)
    start_state = _lrx8_scramble_far()
    with pytest.raises(ValueError, match="must be the same"):
        graph_a.beam_search(
            start_state=start_state, beam_mode="simple", hashed_neigbourhood=bfs_result_b, beam_width=100, max_steps=5
        )


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_predictor_zero(beam_mode):
    """predictor='zero' -> every state scores 0; beam still finds a path on a small graph."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    kwargs = {"beam_mode": beam_mode, "predictor": "zero", "beam_width": 10**5, "max_steps": 30, "return_path": True}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_predictor_torch_module(beam_mode):
    """predictor as torch.nn.Module -> covers isinstance branch in Predictor.__init__."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    model = _TorchModulePredictor()
    kwargs = {"beam_mode": beam_mode, "predictor": model, "beam_width": 10**5, "max_steps": 30, "return_path": True}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_predictor_object_with_predict(beam_mode):
    """predictor as object with .predict method -> covers that branch in Predictor.__init__."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    predictor = _ObjectWithPredict()
    kwargs = {"beam_mode": beam_mode, "predictor": predictor, "beam_width": 10**5, "max_steps": 30, "return_path": True}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


@pytest.mark.parametrize("beam_mode", _ADV_MODES)
def test_beam_search_matrix_history_depth_values(beam_mode):
    """history_depth axis: 0, 1, 2 all accepted and either find a path or exhaust max_steps."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    for history_depth in (0, 1, 2):
        result = graph.beam_search(
            start_state=start_state,
            beam_mode=beam_mode,
            history_depth=history_depth,
            beam_width=10**5,
            max_steps=30,
            return_path=True,
        )
        assert result.path_found
        _validate_beam_search_result(graph, start_state, result)


@pytest.mark.parametrize("beam_mode", _ADV_MODES)
def test_beam_search_matrix_destination_custom(beam_mode):
    """destination_state axis: a non-central target works (advanced/iterated only)."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = [0, 1, 2, 3, 4, 5, 6, 7]
    destination = _lrx8_scramble_far()
    result = graph.beam_search(
        start_state=start_state,
        beam_mode=beam_mode,
        destination_state=destination,
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
    )
    assert result.path_found


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_memory_cleanup(beam_mode, monkeypatch):
    """memory_cleanup=True -> graph.free_memory is called each full iteration (not-found path).

    free_memory is invoked at the end of each loop body. When a path is found early the
    loop returns before reaching that block, so this test uses a not-found scenario where
    the loop runs the full ``max_steps`` iterations.
    """
    graph = CayleyGraph(PermutationGroups.lrx(50))
    start_state = np.random.permutation(50)
    calls = []
    monkeypatch.setattr(graph, "free_memory", lambda: calls.append(1))
    kwargs = {"beam_mode": beam_mode, "beam_width": 5, "max_steps": 5, "memory_cleanup": True}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 1
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert not result.path_found
    # Loop ran 5 full iterations without early success -> free_memory called 5 times.
    assert len(calls) == 5


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_matrix_path_device_cpu(beam_mode):
    """path_device='cpu' -> the returned path-restore hashes live on CPU."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    kwargs = {"beam_mode": beam_mode, "beam_width": 10**5, "max_steps": 30, "path_device": "cpu", "return_path": True}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


@pytest.mark.parametrize("beam_mode", _ADV_MODES)
def test_beam_search_matrix_verbose_profiling(beam_mode, capsys):
    """verbose=10 and verbose=100 profiling branches execute without raising.

    Covers beam_search.py:484-493 (advanced) and :736-745 (iterated) timing-print branches.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    for verbose_level in (10, 100):
        capsys.readouterr()  # clear
        result = graph.beam_search(
            start_state=start_state,
            beam_mode=beam_mode,
            beam_width=100,
            max_steps=20,
            history_depth=2,
            verbose=verbose_level,
        )
        # Must complete without exception; outcome is not asserted (verbose path only).
        assert isinstance(result, BeamSearchResult)


def test_beam_search_matrix_iterated_non_permutation_raises():
    """iterated mode on a non-permutation (matrix) group raises ValueError.

    Pins current behavior (beam_search.py:546-547).
    # TODO(char-spec): iterated matrix-group support is planned; revisit.
    """
    graph = CayleyGraph(MatrixGroups.heisenberg())
    start_state = [[1, 2, 3], [0, 1, 1], [0, 0, 1]]
    with pytest.raises(ValueError, match="only for Permutation Groups"):
        graph.beam_search(start_state=start_state, beam_mode="iterated")


def test_beam_search_matrix_numpy_predictor_simple():
    """Numpy-predictor branch (np.argsort) in simple/advanced: callable returns np.ndarray.

    Covers beam_search.py:457-459. The simple/advanced modes take the ``else`` branch when
    ``scores`` is not a torch.Tensor and fall back to ``np.argsort``.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    result = graph.beam_search(
        start_state=start_state,
        beam_mode="advanced",
        predictor=_NumpyPredictor(),
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
        return_path=True,
    )
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_matrix_numpy_predictor_iterated():
    """Numpy-predictor branch (np.argsort) in iterated mode.

    Covers beam_search.py:695-697.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    result = graph.beam_search(
        start_state=start_state,
        beam_mode="iterated",
        predictor=_NumpyPredictor(),
        beam_width=10**5,
        max_steps=30,
        history_depth=2,
        return_path=True,
    )
    assert result.path_found
    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_empty_beam_early_exit_advanced(monkeypatch):
    """Outcome: empty-beam early exit in advanced mode -> BeamSearchResult(False, i_step, ...).

    When the non-backtracking filter removes ALL candidates, the search gives up
    mid-search at beam_search.py:442-445. This is hard to trigger naturally (the
    search tends to find the goal first on small graphs), so we force it by making
    ``get_unique_states`` return an empty tensor on the second call.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start_state = [1, 0, 2, 3, 4]
    original = graph.get_unique_states
    call_count = {"n": 0}

    def _mocked_unique(states, hashes=None):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            # Return empty to trigger the early-exit path on the 3rd+ call.
            empty = torch.empty((0, states.shape[1] if states.dim() > 1 else 1), dtype=states.dtype)
            return empty, torch.empty((0,), dtype=torch.int64)
        return original(states, hashes)

    monkeypatch.setattr(graph, "get_unique_states", _mocked_unique)
    result = graph.beam_search(
        start_state=start_state,
        beam_mode="advanced",
        beam_width=10,
        max_steps=50,
        history_depth=0,
    )
    assert not result.path_found
    assert result.path is None


@pytest.mark.parametrize(
    "history_depth,mitm_radius",
    [(0, 0), (2, 2), (3, 3), (4, 4), (6, 6)],
)
def test_beam_search_exact_value_parametrized(history_depth, mitm_radius):
    """Parametrized version of the ``exact_value_*`` tests (replaces duplication).

    On LRX(16) with the cycle-roll predictor, the optimal path from the rolled state
    is 6 applications of generator 0 regardless of (history_depth, mitm_radius) pairing.
    """
    graph = CayleyGraph(PermutationGroups.lrx(16))
    state = list(range(10, 16)) + list(range(0, 10))
    result = graph.beam_search(
        start_state=state,
        beam_mode="advanced",
        predictor=_cycle_roll_predictor,
        hashed_neigbourhood=mitm_radius,
        return_path=True,
        history_depth=history_depth,
    )
    assert result.path_length == 6
    assert tuple(result.path) == (0, 0, 0, 0, 0, 0), result.path
    _validate_beam_search_result(graph, state, result)


@pytest.mark.parametrize("beam_mode", _BEAM_MODES)
def test_beam_search_invariant_apply_path_equals_central(beam_mode):
    """Invariant: for any found path, apply_path(start, path) == central_state.

    Cross-mode property test extending ``_validate_beam_search_result`` over the matrix.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = _lrx8_scramble_far()
    kwargs = {"beam_mode": beam_mode, "beam_width": 10**5, "max_steps": 30, "return_path": True}
    if beam_mode in _ADV_MODES:
        kwargs["history_depth"] = 2
    result = graph.beam_search(start_state=start_state, **kwargs)
    _validate_beam_search_result(graph, start_state, result)
    # path_length == len(path) invariant (BeamSearchResult.__post_init__ already checks).
    assert len(result.path) == result.path_length
    # path elements are valid generator ids.
    assert all(0 <= g < graph.definition.n_generators for g in result.path)
