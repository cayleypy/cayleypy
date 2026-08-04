"""Tests for beam search algorithm."""

import itertools
import os

import numpy as np
import pytest
import torch

from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import CayleyGraphDef
from ..graphs_lib import PermutationGroups, MatrixGroups, prepare_graph
from ..lower_bound import BfsLowerBound
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
    # Scores are recorded on every step where the beam was truncated, so this compares whole beams, not only the
    # answer. Hamming distance is computed in integer arithmetic, so scores must match exactly.
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
# Tests for non-backtracking in "simple" beam search mode
# =============================================================================


def _expand_with_non_backtracking(graph: CayleyGraph, states: torch.Tensor):
    """Expands `states` once without bans, then expands the result banning moves undoing the previous move."""
    inverse_map = graph.definition.generators_inverse_map
    assert inverse_map is not None
    layer1 = _expand_layer(graph, states)
    banned_moves = torch.tensor(inverse_map)[layer1.moves]
    return layer1, _expand_layer(graph, layer1.states), _expand_layer(graph, layer1.states, banned_moves)


def test_expand_layer_bans_moves():
    """Test that banned generators are not applied, and provenance of the remaining states stays correct."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    parents = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [2, 3, 4, 0, 1]]
    banned_moves = [0, 1, 2]

    expanded = _expand_layer(graph, graph.encode_states(parents), torch.tensor(banned_moves))

    expected = {
        tuple(graph.apply_path(parents[p], [g]).reshape((-1)).tolist())
        for p in range(len(parents))
        for g in range(3)
        if g != banned_moves[p]
    }
    assert {tuple(x) for x in graph.decode_states(expanded.states).tolist()} == expected
    assert len(expanded.states) == len(expected)
    for i in range(len(expanded.states)):
        parent_id = int(expanded.source_index[i]) % len(parents)
        assert int(expanded.moves[i]) != banned_moves[parent_id]


def test_expand_layer_keeps_state_reachable_by_allowed_move():
    """Test that a state is banned only along the banned move, and survives if another move also produces it."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    # X applied to the first state and L applied to the second state both give [1, 0, 2, 3, 4].
    parents = graph.encode_states([[0, 1, 2, 3, 4], [4, 1, 0, 2, 3]])

    expanded = _expand_layer(graph, parents, torch.tensor([2, 2]))

    states = [tuple(x) for x in graph.decode_states(expanded.states).tolist()]
    # [1, 0, 2, 3, 4] is still there, produced by generator 0 (L) from the second state (index 1 in the layer).
    assert (1, 0, 2, 3, 4) in states
    i = states.index((1, 0, 2, 3, 4))
    assert int(expanded.moves[i]) == 0
    assert int(expanded.source_index[i]) == 1
    # [1, 4, 0, 2, 3] was produced only by the banned move (X applied to the second state), so it is gone.
    assert (1, 4, 0, 2, 3) not in states
    assert (1, 4, 0, 2, 3) in [tuple(x) for x in graph.decode_states(_expand_layer(graph, parents).states).tolist()]


def test_expand_layer_banned_moves_remove_previous_layer():
    """Test that the states removed by banning are exactly the states of the previous layer."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    start_state = (0, 1, 2, 3, 4)

    _, plain, non_backtracking = _expand_with_non_backtracking(graph, graph.encode_states(list(start_state)))

    plain_states = {tuple(x) for x in graph.decode_states(plain.states).tolist()}
    nbt_states = {tuple(x) for x in graph.decode_states(non_backtracking.states).tolist()}
    # Undoing the move that produced a state leads exactly to the state it was produced from.
    assert plain_states - nbt_states == {start_state}


def test_expand_layer_banned_moves_with_involutions():
    """Test banning when every generator is its own inverse, so the banned move is the move itself."""
    graph_def = PermutationGroups.cyclic_coxeter(5)
    graph = CayleyGraph(graph_def, device="cpu")
    assert graph_def.generators_inverse_map == [0, 1, 2, 3, 4]
    start_state = (0, 1, 2, 3, 4)

    layer1, plain, non_backtracking = _expand_with_non_backtracking(graph, graph.encode_states(list(start_state)))

    assert len(layer1.states) == 5
    # Applying the same transposition twice returns to the start state, and only that state is banned.
    assert len(plain.states) == 16 and len(non_backtracking.states) == 15
    assert start_state not in {tuple(x) for x in graph.decode_states(non_backtracking.states).tolist()}


def test_beam_search_simple_non_backtracking_solves_more_states():
    """Test that with a narrow beam, banning moves that undo the previous move solves much more states."""
    n = 5
    graph = CayleyGraph(PermutationGroups.lrx(n), device="cpu")
    all_states = [list(p) for p in itertools.permutations(range(n))]

    lengths = {}
    for non_backtracking in [False, True]:
        results = [
            graph.beam_search(start_state=s, beam_width=1, max_steps=30, non_backtracking=non_backtracking)
            for s in all_states
        ]
        lengths[non_backtracking] = [r.path_length if r.path_found else None for r in results]

    # With beam width 1, the plain search can only follow the Hamming heuristic downhill and gets stuck almost
    # immediately, while the non-backtracking search is forced to explore.
    assert sum(x is not None for x in lengths[False]) < 15
    assert sum(x is not None for x in lengths[True]) > 40
    # No state solved by the plain search is solved worse (or not at all) by the non-backtracking one.
    for plain_length, nbt_length in zip(lengths[False], lengths[True]):
        if plain_length is not None:
            assert nbt_length is not None and nbt_length <= plain_length


def test_beam_search_simple_non_backtracking_finds_valid_path():
    """Test that the path found with non-backtracking is a valid path."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]

    result = graph.beam_search(
        start_state=start_state, beam_width=10, max_steps=50, return_path=True, non_backtracking=True
    )

    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_simple_non_backtracking_disabled_by_default():
    """Test that non_backtracking=False is exactly the old behavior."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]

    result1 = graph.beam_search(start_state=start_state, beam_width=10, max_steps=50, return_path=True)
    result2 = graph.beam_search(
        start_state=start_state, beam_width=10, max_steps=50, return_path=True, non_backtracking=False
    )

    _validate_beam_search_result(graph, start_state, result1)
    assert result1.path == result2.path
    assert len(result1.debug_scores) > 0
    assert result1.debug_scores == result2.debug_scores


def test_beam_search_simple_non_backtracking_with_child_scores():
    """Test that banned moves keep scores of children matched with the states they belong to."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    kwargs = {"beam_width": 10, "max_steps": 50, "return_path": True, "non_backtracking": True}

    result_scalar = graph.beam_search(start_state=start_state, predictor=Predictor(graph, "hamming"), **kwargs)
    result_q = graph.beam_search(
        start_state=start_state, predictor=_ChildrenHammingPredictor(graph), use_child_scores=True, **kwargs
    )

    _validate_beam_search_result(graph, start_state, result_q)
    assert result_scalar.path == result_q.path
    assert len(result_scalar.debug_scores) > 0
    assert result_scalar.debug_scores == result_q.debug_scores


def test_beam_search_simple_non_backtracking_with_canonical_dedup():
    """Test that banned moves stay matched with their states when the beam is deduplicated by symmetries."""
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    kwargs = {
        "beam_width": 10,
        "max_steps": 50,
        "return_path": True,
        "non_backtracking": True,
        "canonical_dedup": symmetry_group,
    }

    result_scalar = graph.beam_search(start_state=start_state, predictor=Predictor(graph, "hamming"), **kwargs)
    result_q = graph.beam_search(
        start_state=start_state, predictor=_ChildrenHammingPredictor(graph), use_child_scores=True, **kwargs
    )

    # Both deduplication by symmetries and banning of moves reorder and filter the expanded layer. If moves or score
    # indexes were not filtered along with the states, they would be assigned to wrong states.
    _validate_beam_search_result(graph, start_state, result_scalar)
    assert result_scalar.path == result_q.path
    assert len(result_scalar.debug_scores) > 0
    assert result_scalar.debug_scores == result_q.debug_scores


def test_beam_search_simple_non_backtracking_with_meet_in_the_middle():
    """Test that non-backtracking works together with meet-in-the-middle."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    bfs_result = graph.bfs(max_diameter=3, return_all_hashes=True)

    result = graph.beam_search(
        start_state=start_state,
        beam_width=10,
        max_steps=50,
        return_path=True,
        bfs_result_for_mitm=bfs_result,
        non_backtracking=True,
    )

    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_simple_non_backtracking_all_moves_banned():
    """Test that the search stops when every move from every state of the beam is banned."""
    # The only generator is its own inverse, so from the second step on there is nowhere to go.
    graph_def = CayleyGraphDef.create([[1, 0, 2, 3]], generator_names=["S"], central_state=[0, 1, 2, 3])
    graph = CayleyGraph(graph_def, device="cpu")
    start_state = [0, 1, 3, 2]

    result = graph.beam_search(start_state=start_state, beam_width=1, max_steps=10, non_backtracking=True)
    plain_result = graph.beam_search(start_state=start_state, beam_width=1, max_steps=10)

    assert not result.path_found
    # The central state is not reachable at all, but the plain search keeps oscillating until it runs out of steps.
    assert not plain_result.path_found
    assert len(result.debug_scores) == 1
    assert len(plain_result.debug_scores) == 10


def test_beam_search_simple_non_backtracking_requires_inverse_closed_generators():
    """Test that non-backtracking is rejected when the inverse of a generator is not a generator."""
    graph_def = PermutationGroups.lx(5)
    assert graph_def.generators_inverse_map is None
    graph = CayleyGraph(graph_def, device="cpu")

    with pytest.raises(ValueError, match="inverse-closed"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], non_backtracking=True)


def test_beam_search_non_backtracking_not_supported_in_advanced_mode():
    """Test that non-backtracking is rejected in "advanced" mode, which has history_depth for this."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    with pytest.raises(ValueError, match="only in 'simple' beam mode"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="advanced", non_backtracking=True)


# =============================================================================
# Tests for lower bound pruning in "simple" beam search mode
# =============================================================================


def _exact_distances(graph: CayleyGraph) -> dict[tuple[int, ...], int]:
    """Returns exact distances from all states to the central state, computed by BFS."""
    bfs_result = graph.bfs(max_layer_size_to_store=None)
    return {tuple(int(x) for x in s): d for d, layer in bfs_result.layers.items() for s in layer}


def _exact_lower_bound(graph: CayleyGraph) -> BfsLowerBound:
    """Returns lower bound that is exact for every state of the graph (full BFS)."""
    return BfsLowerBound(graph, graph.bfs(return_all_hashes=True))


class _WrongShapeLowerBound:
    """Lower bound returning one value for the whole batch instead of one value per state."""

    def lb(self, states: torch.Tensor) -> torch.Tensor:
        assert states.shape[0] > 0
        return torch.zeros((1,), dtype=torch.int64)


def test_beam_search_lower_bound_with_exact_bound_finds_optimal_paths():
    """Test that with an exact lower bound, the beam keeps only states on optimal paths.

    A state reached in `k` steps is kept only if `k+lb(state) <= prune_above`. When the bound is exact and
    `prune_above` is the true distance, this means the state is on some optimal path. So beam search finds an optimal
    path from every state, even with beam width 1 (where it otherwise follows the Hamming heuristic and gets stuck).
    """
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    lower_bound = _exact_lower_bound(graph)
    solved_plain, solved_pruned = 0, 0

    for state, distance in _exact_distances(graph).items():
        if distance == 0:
            continue
        kwargs = {"start_state": list(state), "beam_width": 1, "max_steps": 30, "return_path": True}
        plain = graph.beam_search(**kwargs)
        pruned = graph.beam_search(lower_bound=lower_bound, prune_above=distance, **kwargs)
        solved_plain += plain.path_found
        assert pruned.path_found, f"No path found from {state}, but the optimal path was not pruned."
        assert pruned.path_length == distance
        _validate_beam_search_result(graph, list(state), pruned)
        solved_pruned += 1

    assert solved_plain < 15
    assert solved_pruned == 119


def test_beam_search_lower_bound_does_not_change_result_when_bound_is_not_binding():
    """Test that pruning with a large `prune_above` changes nothing (nothing can be pruned)."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=3, return_all_hashes=True))
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    kwargs = {"start_state": start_state, "beam_width": 10, "max_steps": 50, "return_path": True}

    result1 = graph.beam_search(**kwargs)
    result2 = graph.beam_search(lower_bound=lower_bound, prune_above=1000, **kwargs)

    _validate_beam_search_result(graph, start_state, result2)
    assert result1.path == result2.path
    assert len(result1.debug_scores) > 0
    assert result1.debug_scores == result2.debug_scores


def test_beam_search_lower_bound_disabled_by_default():
    """Test that without `lower_bound` and `prune_above`, the search is exactly the old one."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    kwargs = {"start_state": start_state, "beam_width": 10, "max_steps": 50, "return_path": True}

    result1 = graph.beam_search(**kwargs)
    result2 = graph.beam_search(lower_bound=None, prune_above=None, **kwargs)

    assert result1.path == result2.path
    assert len(result1.debug_scores) > 0
    assert result1.debug_scores == result2.debug_scores


def test_beam_search_prune_above_below_distance_fails_immediately():
    """Test that the search stops as soon as the lower bound says the budget cannot be met."""
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    lower_bound = _exact_lower_bound(graph)
    start_state = [5, 4, 3, 2, 1, 0]
    distance = _exact_distances(graph)[tuple(start_state)]
    assert distance == 13

    result = graph.beam_search(
        start_state=start_state, beam_width=100, max_steps=50, lower_bound=lower_bound, prune_above=distance
    )

    assert result.path_found and result.path_length == distance
    for prune_above in range(distance):
        pruned = graph.beam_search(
            start_state=start_state, beam_width=100, max_steps=50, lower_bound=lower_bound, prune_above=prune_above
        )
        assert not pruned.path_found
        # All states of the layer are pruned as soon as the remaining budget is less than the distance left to go.
        assert len(pruned.debug_scores) <= prune_above


def test_beam_search_prune_above_does_not_reject_a_found_path():
    """Test that a path is not lost because it is longer than the budget: pruning happens after the path is found."""
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    lower_bound = _exact_lower_bound(graph)
    kwargs = {"beam_width": 10, "max_steps": 50, "lower_bound": lower_bound, "prune_above": 0}

    # The central state is one step away, and reaching it is noticed before anything is pruned.
    result1 = graph.beam_search(start_state=[1, 0, 2, 3, 4, 5], **kwargs)
    # Here it is two steps away, and everything is pruned after the first step.
    result2 = graph.beam_search(start_state=[1, 2, 0, 3, 4, 5], **kwargs)

    assert result1.path_found and result1.path_length == 1
    assert not result2.path_found


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_beam_search_lower_bound_from_partial_bfs_gets_stronger_with_radius():
    """Test that a bound from a larger ball prunes more, so more states are solved optimally."""
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    distances = _exact_distances(graph)
    solved = {}

    for radius in [0, 3, 9]:
        lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=radius, return_all_hashes=True))
        solved[radius] = 0
        for state, distance in distances.items():
            if distance == 0:
                continue
            result = graph.beam_search(
                start_state=list(state),
                beam_width=3,
                max_steps=30,
                lower_bound=lower_bound,
                prune_above=distance,
            )
            # Every path found under this bound has optimal length, because longer paths are pruned.
            assert not result.path_found or result.path_length == distance
            solved[radius] += result.path_found

    # With radius 0, the bound is "0 for the central state, 1 for everything else" - almost nothing is pruned.
    assert solved[0] < 50
    assert solved[3] > 100
    assert solved[9] > 600


def test_beam_search_lower_bound_with_child_scores_dedup_and_non_backtracking():
    """Test that pruning keeps scores and moves of children matched with the states they belong to."""
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, device="cpu")
    lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=3, return_all_hashes=True))
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]
    kwargs = {
        "start_state": start_state,
        "beam_width": 10,
        "max_steps": 50,
        "return_path": True,
        "lower_bound": lower_bound,
        "prune_above": 12,
        "canonical_dedup": SymmetryGroup.reflections(graph_def),
        "non_backtracking": True,
    }

    result_scalar = graph.beam_search(predictor=Predictor(graph, "hamming"), **kwargs)
    result_q = graph.beam_search(predictor=_ChildrenHammingPredictor(graph), use_child_scores=True, **kwargs)

    _validate_beam_search_result(graph, start_state, result_scalar)
    assert result_scalar.path_length <= 12
    assert result_scalar.path == result_q.path
    assert len(result_scalar.debug_scores) > 0
    assert result_scalar.debug_scores == result_q.debug_scores


def test_beam_search_lower_bound_with_meet_in_the_middle():
    """Test that pruning works together with meet-in-the-middle."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    bfs_result = graph.bfs(max_diameter=3, return_all_hashes=True)
    lower_bound = BfsLowerBound(graph, bfs_result)
    start_state = [3, 0, 2, 4, 5, 6, 7, 1]

    result = graph.beam_search(
        start_state=start_state,
        beam_width=10,
        max_steps=50,
        return_path=True,
        bfs_result_for_mitm=bfs_result,
        lower_bound=lower_bound,
        prune_above=12,
    )

    _validate_beam_search_result(graph, start_state, result)


def test_beam_search_lower_bound_requires_prune_above():
    """Test that a lower bound without a budget to compare it with is rejected."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    lower_bound = _exact_lower_bound(graph)

    with pytest.raises(ValueError, match="must be specified together"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], lower_bound=lower_bound)


def test_beam_search_prune_above_requires_lower_bound():
    """Test that a budget without a lower bound to compare with it is rejected."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    with pytest.raises(ValueError, match="must be specified together"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], prune_above=10)


def test_beam_search_prune_above_must_be_non_negative():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    lower_bound = _exact_lower_bound(graph)

    with pytest.raises(ValueError, match="non-negative"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], lower_bound=lower_bound, prune_above=-1)


def test_beam_search_lower_bound_rejects_wrong_shape():
    """Test that a lower bound returning one value for the whole batch is rejected."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    with pytest.raises(ValueError, match="shape"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], lower_bound=_WrongShapeLowerBound(), prune_above=10)


def test_beam_search_lower_bound_not_supported_in_advanced_mode():
    """Test that pruning is rejected in "advanced" mode, where it is not implemented."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    lower_bound = _exact_lower_bound(graph)

    with pytest.raises(ValueError, match="only in 'simple' beam mode"):
        graph.beam_search(start_state=[4, 1, 0, 2, 3], beam_mode="advanced", lower_bound=lower_bound, prune_above=10)


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
