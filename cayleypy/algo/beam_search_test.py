"""Tests for beam search algorithm."""

import os

import numpy as np
import pytest
import torch

from ..cayley_graph import CayleyGraph

from ..graphs_lib import PermutationGroups, MatrixGroups, prepare_graph
from ..predictor import Predictor
from ..puzzles import Puzzles
from ..symmetries import SymmetryGroup
from .beam_search import _dedup_by_canonical_form, _expand_layer, _score_children
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


class _ChildrenHammingModel(torch.nn.Module):
    """Emulates a Q-model: returns Hamming distances of all children of a state, one output per generator."""

    def __init__(self, graph: CayleyGraph):
        super().__init__()
        self.permutations = torch.as_tensor(graph.definition.generators_permutations, device=graph.device)
        self.central_state = graph.central_state

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # children[i, j, :] is the state obtained by applying generator j to states[i].
        children = states[:, self.permutations]
        return torch.sum(children != self.central_state, dim=2)


class _ChildrenHammingPredictor(Predictor):
    """Predictor for a model that has one output per generator."""

    def __init__(self, graph: CayleyGraph):
        super().__init__(graph, _ChildrenHammingModel(graph))

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        return self.predict_batched(states)


class _BrokenChildrenPredictor(Predictor):
    """Predictor whose model has wrong number of outputs (one more than the number of generators)."""

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        return torch.zeros((states.shape[0], self.graph.definition.n_generators + 1))


class _OrbitRecordingPredictor(Predictor):
    """Hamming distance predictor that remembers how many distinct orbits were among the states it scored."""

    def __init__(self, graph: CayleyGraph, symmetry_group: SymmetryGroup):
        self.symmetry_group = symmetry_group
        self.layers: list[tuple[int, int]] = []
        super().__init__(graph, self._predict)

    def _predict(self, states: torch.Tensor) -> torch.Tensor:
        canonical = self.symmetry_group.canonical(states)
        self.layers.append((int(states.shape[0]), len({tuple(x.tolist()) for x in canonical})))
        return torch.sum(states != self.graph.central_state, dim=1)


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
        start_state=state, beam_mode="simple", predictor=predictor, bfs_result_for_mitm=bfs_result, return_path=True
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
# Tests for child scoring in "simple" beam search mode
# =============================================================================


def test_expand_layer_keeps_provenance():
    """Test that layer expansion deduplicates as usual, but remembers which generator produced each state."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    parents = graph.encode_states([[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [2, 3, 4, 0, 1]])
    n_parents = int(parents.shape[0])

    expanded = _expand_layer(graph, parents)

    expected_states, expected_hashes = graph.get_unique_states(graph.get_neighbors(parents))
    assert torch.equal(expanded.states, expected_states)
    assert torch.equal(expanded.hashes, expected_hashes)

    # Applying generator `moves[i]` to the state that produced i-th state must give exactly that state.
    decoded_parents = graph.decode_states(parents)
    assert len(expanded.moves) == len(expanded.states)
    for i in range(len(expanded.states)):
        parent_id = int(expanded.source_index[i]) % n_parents
        child = graph.apply_path(decoded_parents[parent_id], [int(expanded.moves[i])])
        assert torch.equal(graph.encode_states(child), expanded.states[i : i + 1])


def test_expand_layer_empty_frontier():
    """Test that expanding and scoring an empty layer gives empty answers rather than an error."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    no_states = torch.zeros((0, graph.encoded_state_size), dtype=torch.int64)

    expanded = _expand_layer(graph, no_states)
    assert expanded.states.shape == (0, graph.encoded_state_size)
    assert len(expanded.hashes) == 0
    assert len(expanded.moves) == 0
    assert len(expanded.source_index) == 0

    assert len(_score_children(graph, _ChildrenHammingPredictor(graph), no_states)) == 0


def test_beam_search_simple_child_scores_same_as_default():
    """Test that child scoring with the same predictor gives exactly the same beam."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]

    result1 = graph.beam_search(start_state=start_state, beam_width=10, max_steps=50, return_path=True)
    result2 = graph.beam_search(
        start_state=start_state, beam_width=10, max_steps=50, return_path=True, use_child_scores=True
    )

    _validate_beam_search_result(graph, start_state, result1)
    assert result1.path == result2.path
    # One score is recorded per step where the beam was truncated - the best score of that step - so this compares the
    # two searches step by step, not only their answers. Hamming distance is computed in integer arithmetic, so the
    # scores must match exactly.
    assert len(result1.debug_scores) > 0
    assert result1.debug_scores == result2.debug_scores


def test_beam_search_simple_child_scores_same_as_default_when_path_not_found():
    """Test that child scoring does not change the beam even on a long unsuccessful search."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [7, 6, 5, 4, 3, 2, 1, 0]

    result1 = graph.beam_search(start_state=start_state, beam_width=20, max_steps=50)
    result2 = graph.beam_search(start_state=start_state, beam_width=20, max_steps=50, use_child_scores=True)

    assert not result1.path_found
    assert not result2.path_found
    assert len(result1.debug_scores) > 40
    assert result1.debug_scores == result2.debug_scores


def test_beam_search_simple_child_scores_q_model_parity():
    """Test that a model with one output per generator gives the same beam as its scalar equivalent."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]

    result_scalar = graph.beam_search(
        start_state=start_state, predictor=Predictor(graph, "hamming"), beam_width=10, max_steps=50, return_path=True
    )
    result_q = graph.beam_search(
        start_state=start_state,
        predictor=_ChildrenHammingPredictor(graph),
        beam_width=10,
        max_steps=50,
        return_path=True,
        use_child_scores=True,
    )

    _validate_beam_search_result(graph, start_state, result_q)
    assert result_scalar.path == result_q.path
    assert result_scalar.debug_scores == result_q.debug_scores


def test_beam_search_simple_child_scores_wrong_number_of_outputs():
    """Test that a model with wrong number of outputs is reported clearly."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    predictor = _BrokenChildrenPredictor(graph, "zero")

    with pytest.raises(ValueError, match="one output per generator"):
        graph.beam_search(
            start_state=[7, 6, 5, 4, 3, 2, 1, 0], beam_width=3, max_steps=50, use_child_scores=True, predictor=predictor
        )


def test_beam_search_child_scores_not_supported_in_advanced_mode():
    """Test that child scoring is rejected in "advanced" mode, where it is not implemented."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    with pytest.raises(ValueError, match="only in 'simple' beam mode"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="advanced", use_child_scores=True)


# =============================================================================
# Tests for deduplication by symmetries in "simple" beam search mode
# =============================================================================


def test_dedup_by_canonical_form_keeps_one_state_per_orbit():
    """Test that deduplication keeps exactly one state from each orbit, on a hand-computed example."""
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    # The reflection maps [0, 2, 1, 3, 4] to [4, 1, 2, 3, 0] and [2, 0, 1, 3, 4] to [1, 4, 2, 3, 0], and it preserves
    # the identity permutation. So, these 5 states form 3 orbits.
    states = graph.encode_states([[0, 2, 1, 3, 4], [4, 1, 2, 3, 0], [2, 0, 1, 3, 4], [1, 4, 2, 3, 0], [0, 1, 2, 3, 4]])

    keep = _dedup_by_canonical_form(graph, states, symmetry_group)

    assert len(keep) == 3
    kept_states = graph.decode_states(states[keep, :])
    canonical_kept = [tuple(x.tolist()) for x in symmetry_group.canonical(kept_states)]
    # Exactly one state from each of the 3 orbits was kept.
    assert len(set(canonical_kept)) == 3
    assert set(canonical_kept) == {(0, 2, 1, 3, 4), (1, 4, 2, 3, 0), (0, 1, 2, 3, 4)}


def test_dedup_by_canonical_form_compresses_bfs_layer():
    """Test deduplication of a set of states that is closed under symmetries: a whole BFS layer."""
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    layer = graph.bfs(max_layer_size_to_store=None, max_diameter=4).layers[4]
    assert layer.shape[0] == 6539

    keep = _dedup_by_canonical_form(graph, graph.encode_states(layer), symmetry_group)

    # Rotations of the whole cube preserve distances, so this layer consists of whole orbits. There are 24 rotations,
    # hence the layer shrinks by almost 24 times (exactly 24 for states that no rotation preserves).
    assert len(keep) == len({tuple(x.tolist()) for x in symmetry_group.canonical(layer)}) == 294


def test_beam_search_simple_canonical_dedup_finds_path_where_plain_search_fails():
    """Test that deduplication by symmetries makes the beam cover more distinct states."""
    graph_def = PermutationGroups.lrx(6)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    start_state = [5, 4, 3, 2, 1, 0]

    predictor = _OrbitRecordingPredictor(graph, symmetry_group)
    result = graph.beam_search(
        start_state=start_state,
        predictor=predictor,
        beam_width=200,
        max_steps=20,
        return_path=True,
        canonical_dedup=symmetry_group,
    )
    plain_predictor = _OrbitRecordingPredictor(graph, symmetry_group)
    plain_result = graph.beam_search(
        start_state=start_state, predictor=plain_predictor, beam_width=200, max_steps=20, return_path=True
    )

    # Almost half of the states of the plain beam are equivalent to some other state in it, so it hits the beam width
    # and gets truncated - and then the (weak) Hamming heuristic never recovers the path.
    assert min(orbits / states for states, orbits in plain_predictor.layers) < 0.6
    assert not plain_result.path_found
    # With deduplication, the same beam width is enough to hold all distinct orbits, and the path is found.
    _validate_beam_search_result(graph, start_state, result)
    assert result.path_length == 13


def test_beam_search_simple_canonical_dedup_leaves_no_equivalent_states_in_beam():
    """Test that with deduplication, no two states scored on the same step are equivalent under symmetries."""
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    start_state = [7, 6, 5, 4, 3, 2, 1, 0]

    predictor = _OrbitRecordingPredictor(graph, symmetry_group)
    graph.beam_search(
        start_state=start_state, predictor=predictor, beam_width=300, max_steps=25, canonical_dedup=symmetry_group
    )
    plain_predictor = _OrbitRecordingPredictor(graph, symmetry_group)
    graph.beam_search(start_state=start_state, predictor=plain_predictor, beam_width=300, max_steps=25)

    assert len(predictor.layers) > 10
    assert all(states == orbits for states, orbits in predictor.layers)
    # Without deduplication, every one of these layers contains states equivalent to other states in the same layer.
    assert all(orbits < states for states, orbits in plain_predictor.layers)


def test_beam_search_simple_canonical_dedup_with_child_scores():
    """Test that deduplication keeps scores of children matched with the states they belong to."""
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    kwargs = {"beam_width": 10, "max_steps": 50, "return_path": True, "canonical_dedup": symmetry_group}

    result_scalar = graph.beam_search(start_state=start_state, predictor=Predictor(graph, "hamming"), **kwargs)
    result_q = graph.beam_search(
        start_state=start_state, predictor=_ChildrenHammingPredictor(graph), use_child_scores=True, **kwargs
    )

    _validate_beam_search_result(graph, start_state, result_q)
    # Scores of children are looked up by index in the expanded layer, and deduplication reorders that layer. If the
    # indexes were not updated, scores would be assigned to wrong states and the beams would diverge.
    assert result_scalar.path == result_q.path
    assert len(result_scalar.debug_scores) > 0
    assert result_scalar.debug_scores == result_q.debug_scores


def test_beam_search_simple_canonical_dedup_with_meet_in_the_middle():
    """Test that deduplication works together with meet-in-the-middle."""
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    bfs_result = graph.bfs(max_diameter=3, return_all_hashes=True)

    result = graph.beam_search(
        start_state=start_state,
        beam_width=10,
        max_steps=50,
        return_path=True,
        bfs_result_for_mitm=bfs_result,
        canonical_dedup=symmetry_group,
    )

    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_simple_canonical_dedup_with_trivial_group():
    """Test that deduplication by the group consisting of the identity alone changes nothing."""
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, device="cpu")
    trivial_group = SymmetryGroup([list(range(8))], graph_def)
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]

    result1 = graph.beam_search(start_state=start_state, beam_width=10, max_steps=50, return_path=True)
    result2 = graph.beam_search(
        start_state=start_state, beam_width=10, max_steps=50, return_path=True, canonical_dedup=trivial_group
    )

    _validate_beam_search_result(graph, start_state, result2)
    assert result1.path == result2.path
    assert result1.debug_scores == result2.debug_scores


def test_beam_search_canonical_dedup_rejects_symmetries_that_are_not_a_group():
    """Test that a set of symmetries that is not a group is rejected instead of silently dropping states.

    Canonical forms of two states are equal if and only if the symmetries form a group. With a set that is not a group,
    states of one orbit can get different canonical forms, so deduplication throws away states that are the only route
    to the central state - and the search reports that there is no path.
    """
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")

    # A permutation of positions that is not a symmetry of this graph (its conjugate of L is not a generator).
    not_a_symmetry = SymmetryGroup([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4]], graph_def)
    with pytest.raises(ValueError, match="conjugate of generator L"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], canonical_dedup=not_a_symmetry)

    # A genuine symmetry, but the set is not closed under composition (it does not contain the identity).
    not_closed = SymmetryGroup([[1, 0, 4, 3, 2]], graph_def)
    with pytest.raises(ValueError, match="Identity permutation"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], canonical_dedup=not_closed)


def test_beam_search_canonical_dedup_rejects_group_for_other_graph():
    """Test that symmetries of another graph are rejected (they would deduplicate wrong states)."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    symmetry_group = SymmetryGroup.reflections(PermutationGroups.lrx(5, k=2))

    with pytest.raises(ValueError, match="different graph"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], canonical_dedup=symmetry_group)


def test_beam_search_canonical_dedup_not_supported_in_advanced_mode():
    """Test that deduplication by symmetries is rejected in "advanced" mode, where it is not implemented."""
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)

    with pytest.raises(ValueError, match="only in 'simple' beam mode"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="advanced", canonical_dedup=symmetry_group)


# =============================================================================
# Tests for "advanced" beam search mode
# =============================================================================


def test_beam_search_advanced_lrx_few_steps():
    """Test advanced beam search on small LRX graph with few steps."""
    graph = CayleyGraph(PermutationGroups.lrx(5))

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
def test_beam_search_simple_cube222_canonical_dedup():
    """Test deduplication by all 24 rotations of the whole cube on a real puzzle."""
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    # Fixed seed: a scramble that happens to land close to the central state is solved in a few steps, and then there
    # are too few layers to compare.
    torch.manual_seed(0)
    start_state = _scramble(graph, 100)

    predictor = _OrbitRecordingPredictor(graph, symmetry_group)
    graph.beam_search(
        start_state=start_state,
        predictor=predictor,
        beam_width=10**4,
        max_steps=10,
        canonical_dedup=symmetry_group,
    )
    plain_predictor = _OrbitRecordingPredictor(graph, symmetry_group)
    graph.beam_search(start_state=start_state, predictor=plain_predictor, beam_width=10**4, max_steps=10)

    assert len(predictor.layers) >= 4
    assert all(states == orbits for states, orbits in predictor.layers)
    # Without deduplication, a third or more of the beam is spent on states equivalent to other states in it.
    assert min(orbits / states for states, orbits in plain_predictor.layers) < 0.7


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


def test_beam_search_advanced_with_mitm_error():
    """Test that advanced mode with bfs_result_for_mitm raises error."""
    graph = CayleyGraph(PermutationGroups.lrx(8))
    start_state = np.random.permutation(8)
    bfs_result = graph.bfs(max_diameter=5, return_all_hashes=True)

    # This should work (bfs_result_for_mitm is ignored in advanced mode)
    result = graph.beam_search(start_state=start_state, beam_mode="advanced", bfs_result_for_mitm=bfs_result)
    assert result.path_found or result.path_length > 0
