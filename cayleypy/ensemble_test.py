import pytest
import torch

from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef
from .ensemble import EnsemblePredictor
from .graphs_lib import PermutationGroups
from .predictor import Predictor

# States used in most tests below. Distances from the central state of lrx(5) (i.e. [0,1,2,3,4]) are 0, 5, 4, 2.
STATES = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [4, 3, 2, 1, 0], [0, 2, 1, 3, 4]])


def _first_element(states: torch.Tensor) -> torch.Tensor:
    """Heuristic returning the first element of the state (used as a second, unrelated member of ensembles)."""
    return states[:, 0].to(torch.float32)


class QLikePredictor(Predictor):
    """Predictor scoring all children in one call, like Q-models do."""

    def __init__(self, graph: CayleyGraph):
        super().__init__(graph, "hamming")
        self.score_children_calls = 0

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        self.score_children_calls += 1
        n_generators = self.graph.definition.n_generators
        return torch.full((states.shape[0], n_generators), 10.0)


def test_ensemble_of_predictions():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    ensemble = EnsemblePredictor([Predictor(graph, "hamming"), Predictor(graph, _first_element)], [0.75, 0.25])

    # Hamming distances are [0,5,4,2], first elements are [0,1,4,0].
    expected = torch.tensor([0.0, 0.75 * 5 + 0.25 * 1, 0.75 * 4 + 0.25 * 4, 0.75 * 2 + 0.25 * 0])
    assert torch.allclose(ensemble(STATES), expected)


def test_ensemble_of_children_scores():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    n_generators = graph.definition.n_generators
    member1, member2 = Predictor(graph, "hamming"), Predictor(graph, _first_element)
    ensemble = EnsemblePredictor([member1, member2], [0.75, 0.25])

    scores = ensemble.score_children(STATES)
    assert scores.shape == (4, n_generators)
    for i in range(n_generators):
        # Column i must contain scores of children obtained by applying generator i.
        children = graph.apply_path(STATES, [i])
        assert torch.allclose(scores[:, i], 0.75 * member1(children) + 0.25 * member2(children))

    # Sanity check that this test can distinguish columns from each other.
    assert len({tuple(scores[:, i].tolist()) for i in range(n_generators)}) > 1


def test_ensemble_of_one_member_equals_that_member():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    member = Predictor(graph, "hamming")
    ensemble = EnsemblePredictor([member], [1.0])
    assert torch.allclose(ensemble(STATES), member(STATES).to(torch.float32))
    assert torch.allclose(ensemble.score_children(STATES), member.score_children(STATES).to(torch.float32))


def test_default_weights_average_members():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    member1, member2 = Predictor(graph, "hamming"), Predictor(graph, _first_element)
    ensemble = EnsemblePredictor([member1, member2])
    assert ensemble.weights == [0.5, 0.5]
    assert torch.allclose(ensemble(STATES), (member1(STATES) + member2(STATES)) / 2)


def test_ensemble_uses_score_children_of_members():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    fast_member = QLikePredictor(graph)
    slow_member = Predictor(graph, "hamming")
    ensemble = EnsemblePredictor([fast_member, slow_member], [1.0, 1.0])

    scores = ensemble.score_children(STATES)
    assert fast_member.score_children_calls == 1
    assert torch.allclose(scores, 10.0 + slow_member.score_children(STATES).to(torch.float32))


def test_ensemble_batches_predictions():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", batch_size=3)
    member1, member2 = Predictor(graph, "hamming"), Predictor(graph, _first_element)
    ensemble = EnsemblePredictor([member1, member2], [0.75, 0.25])

    # There are 7 states and 21 children, so both of them are processed in several batches.
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])
    assert torch.allclose(ensemble(states), 0.75 * member1(states) + 0.25 * member2(states))
    scores = ensemble.score_children(states)
    assert scores.shape == (7, graph.definition.n_generators)
    for i in range(graph.definition.n_generators):
        children = graph.apply_path(states, [i])
        assert torch.allclose(scores[:, i], 0.75 * member1(children) + 0.25 * member2(children))


def test_ensemble_works_in_beam_search():
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu")
    ensemble = EnsemblePredictor([Predictor(graph, "hamming"), Predictor(graph, "zero")], [0.9, 0.1])
    start_state = [7, 6, 5, 4, 3, 2, 1, 0]

    result = graph.beam_search(
        start_state=start_state, predictor=ensemble, beam_mode="simple", beam_width=10**7, return_path=True
    )
    assert result.path_found
    assert result.path_length <= 28
    assert torch.equal(graph.apply_path(start_state, result.path)[0], torch.tensor(graph.definition.central_state))


def test_weights_are_not_normalized():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    member1, member2 = Predictor(graph, "hamming"), Predictor(graph, _first_element)
    ensemble = EnsemblePredictor([member1, member2], [2.0, 3.0])

    expected = 2.0 * member1(STATES) + 3.0 * member2(STATES)
    assert torch.allclose(ensemble(STATES), expected)
    # Ranking is preserved, but the scale is not: weights summing to 5 are not silently divided by 5.
    assert not torch.allclose(ensemble(STATES), expected / 5)


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError, match="at least one member"):
        EnsemblePredictor([])


def test_wrong_number_of_weights_is_rejected():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    members = [Predictor(graph, "hamming"), Predictor(graph, "zero")]
    with pytest.raises(ValueError, match="2 members, but 3 weights"):
        EnsemblePredictor(members, [0.5, 0.3, 0.2])


def test_members_with_different_number_of_generators_are_rejected():
    graph1 = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    graph2 = CayleyGraph(CayleyGraphDef.create([[1, 2, 3, 4, 0]]), device="cpu")
    assert graph1.definition.state_size == graph2.definition.state_size
    with pytest.raises(ValueError, match="same graph"):
        EnsemblePredictor([Predictor(graph1, "hamming"), Predictor(graph2, "hamming")])


def test_members_with_different_state_size_are_rejected():
    graph1 = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    graph2 = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    with pytest.raises(ValueError, match="same graph"):
        EnsemblePredictor([Predictor(graph1, "hamming"), Predictor(graph2, "hamming")])


def test_member_that_is_not_predictor_is_rejected():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    with pytest.raises(TypeError, match="wrapped in Predictor"):
        EnsemblePredictor([Predictor(graph, "hamming"), "hamming"])  # type: ignore[list-item]
