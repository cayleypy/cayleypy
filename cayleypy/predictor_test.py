import numpy as np
import pytest
import torch

from .cayley_graph import CayleyGraph
from .graphs_lib import PermutationGroups
from .predictor import Predictor


class SklearnStyleModel:
    """Model with a `predict` method returning a NumPy array, as an sklearn estimator does."""

    @staticmethod
    def predict(states: torch.Tensor) -> np.ndarray:
        return np.asarray(states[:, 0], dtype=np.float32)


class MultiOutputModel(torch.nn.Module):
    """Model returning one score per generator (i.e. having 2-D output)."""

    def __init__(self, n_outputs: int):
        super().__init__()
        self.n_outputs = n_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1].float() + torch.arange(self.n_outputs, dtype=torch.float32)


def test_hamming_predictor():
    graph_def = PermutationGroups.lrx(5).with_central_state("01001")
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, "hamming")
    states = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 2, 3],
            [0, 1, 1, 1, 1],
        ]
    )
    assert torch.equal(predictor(states), torch.tensor([2, 3, 0, 5, 3, 2]))


def test_predictor_batches_1d_output():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", batch_size=3)
    predictor = Predictor(graph, "hamming")
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])
    assert torch.equal(predictor(states), predictor.predict(states))


def test_predictor_batches_2d_output():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", batch_size=3)
    predictor = Predictor(graph, MultiOutputModel(graph_def.n_generators))
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])

    # Batches of a multi-output model must be concatenated along dimension 0.
    ans = predictor.predict_batched(states)
    assert ans.shape == (7, graph_def.n_generators)
    assert torch.equal(ans, predictor.predict(states))


def test_predictor_rejects_2d_output():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, MultiOutputModel(graph_def.n_generators))
    states = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    with pytest.raises(ValueError, match="score_children"):
        predictor(states)


def test_score_children():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, "hamming")
    states = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [4, 3, 2, 1, 0], [0, 2, 1, 3, 4]])

    scores = predictor.score_children(states)
    assert scores.shape == (4, graph_def.n_generators)
    for i in range(graph_def.n_generators):
        # Column i must contain scores of children obtained by applying generator i.
        expected = predictor(graph.apply_path(states, [i]))
        assert torch.equal(scores[:, i], expected)

    # Sanity check that this test can distinguish columns from each other.
    assert len({tuple(scores[:, i].tolist()) for i in range(graph_def.n_generators)}) > 1


def test_score_children_single_state():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, "hamming")
    scores = predictor.score_children(torch.tensor([0, 2, 1, 3, 4]))
    assert scores.shape == (1, graph_def.n_generators)
    for i in range(graph_def.n_generators):
        assert torch.equal(scores[:, i], predictor(graph.apply_path([0, 2, 1, 3, 4], [i])))


def test_score_children_with_batching():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", batch_size=4)
    predictor = Predictor(graph, "hamming")
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])

    # There are 7*3=21 children, so they are scored in several batches.
    scores = predictor.score_children(states)
    assert scores.shape == (7, graph_def.n_generators)
    for i in range(graph_def.n_generators):
        assert torch.equal(scores[:, i], predictor(graph.apply_path(states, [i])))


@pytest.mark.parametrize("batch_size", [1024, 4])
def test_predictor_converts_output_of_a_model_not_written_in_torch(batch_size):
    """A model with a `predict` method is a supported predictor, and sklearn estimators return NumPy arrays."""
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", batch_size=batch_size)
    predictor = Predictor(graph, SklearnStyleModel())
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])

    assert isinstance(predictor.predict_batched(states), torch.Tensor)
    # score_children applies operations that a NumPy array does not have, so the conversion must happen before them.
    scores = predictor.score_children(states)
    assert scores.shape == (7, graph_def.n_generators)
    for i in range(graph_def.n_generators):
        assert torch.equal(scores[:, i], predictor(graph.apply_path(states, [i])))


def test_score_children_rejects_2d_output():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, MultiOutputModel(graph_def.n_generators))
    with pytest.raises(ValueError, match="score_children"):
        predictor.score_children(torch.tensor([[0, 1, 2, 3, 4]]))


def test_predict_batched_does_not_track_gradients():
    """Test that inference does not build the graph for a backward pass (which would only waste memory)."""
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", batch_size=4)
    model = torch.nn.Sequential(torch.nn.Linear(5, 1), torch.nn.Flatten(0, 1))
    predictor = Predictor(graph, lambda states: model(states.to(torch.float32)))
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])

    # Weights require gradients, so an output that does not is proof that no graph was built.
    assert model[0].weight.requires_grad
    assert not predictor.predict_batched(states).requires_grad
    assert not predictor(states).requires_grad
    assert not predictor.score_children(states).requires_grad
