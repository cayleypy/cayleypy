import pytest
import torch

from .cayley_graph import CayleyGraph
from .graphs_lib import PermutationGroups
from .predictor import Predictor


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


# =============================================================================
# Predictor.__init__ branch coverage (predictor.py:35-48)
# =============================================================================


def test_predictor_zero():
    """``"zero"`` branch (predictor.py:35) -> returns 0 for all states."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    predictor = Predictor(graph, "zero")
    states = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    result = predictor(states)
    assert torch.equal(result, torch.tensor([0.0, 0.0]))


def test_predictor_torch_module():
    """``torch.nn.Module`` branch (predictor.py:39) -> uses the module in eval mode."""

    class _DummyModel(torch.nn.Module):
        def forward(self, states):
            return (states != 0).sum(dim=1).to(torch.float32)

    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    model = _DummyModel()
    predictor = Predictor(graph, model)
    assert model.training is False
    states = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    result = predictor(states)
    assert torch.equal(result, torch.tensor([0.0, 5.0]))


def test_predictor_object_with_predict():
    """Object with ``.predict`` method branch (predictor.py:43)."""

    class _SklearnLike:
        def predict(self, states):
            return (states != 0).sum(dim=1).to(torch.float32)

    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    predictor = Predictor(graph, _SklearnLike())
    states = torch.tensor([[0, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    result = predictor(states)
    assert torch.equal(result, torch.tensor([1.0, 5.0]))


def test_predictor_callable():
    """Callable object branch (predictor.py:45) -> uses ``__call__``."""

    def _callable(states):
        return (states != 0).sum(dim=1).to(torch.float32)

    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    predictor = Predictor(graph, _callable)
    states = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    result = predictor(states)
    assert torch.equal(result, torch.tensor([0.0, 1.0]))


def test_predictor_invalid_input_raises():
    """``else`` branch (predictor.py:48) -> raises ValueError for unsupported type."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    with pytest.raises(ValueError, match="Unable to understand"):
        Predictor(graph, 12345)


# =============================================================================
# Predictor.__call__ batch-splitting (predictor.py:58-63)
# =============================================================================


def test_predictor_call_single_batch():
    """When states fit in one batch (num_batches == 1), predict is called directly."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", batch_size=10**6)
    predictor = Predictor(graph, "hamming")
    states = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    result = predictor(states)
    assert result.shape == (2,)


def test_predictor_call_multi_batch():
    """When states exceed batch_size, they are split and concatenated (predictor.py:61-62)."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", batch_size=3)
    predictor = Predictor(graph, "hamming")
    # 7 states, batch_size=3 -> 3 batches (3,3,1).
    states = torch.tensor(
        [
            [0, 1, 2, 3, 4],  # central -> distance 0
            [0, 0, 0, 0, 0],  # distance 4
            [0, 0, 0, 0, 0],  # distance 4
            [0, 0, 0, 0, 0],  # distance 4
            [0, 0, 0, 0, 0],  # distance 4
            [0, 0, 0, 0, 0],  # distance 4
            [4, 3, 2, 1, 0],  # distance 4 (position 2 matches: 2==2)
        ]
    )
    result = predictor(states)
    assert result.shape == (7,)
    # First state is central -> distance 0; last state -> distance 4.
    assert float(result[0]) == 0.0
    assert float(result[6]) == 4.0


# =============================================================================
# Predictor.pretrained (predictor.py:50-56)
# =============================================================================


def test_predictor_pretrained_unknown_graph_raises():
    """``pretrained`` raises KeyError for a graph without a model (predictor.py:53-54)."""
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    with pytest.raises(KeyError, match="No pretrained model"):
        Predictor.pretrained(graph)
