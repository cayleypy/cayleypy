import os
from dataclasses import replace

import pytest
import torch

from .checkpoint import graph_hash, save_checkpoint
from .models import ModelConfig
from .models_lib import PREDICTOR_MODELS
from .. import prepare_graph, Predictor, CayleyGraph
from ..graphs_lib import PermutationGroups

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def test_loads_predictor_models():
    # Checks that all models can be loaded and successfully return prediction for central state of the graph.
    # This test does not check model quality.
    for graph_name, config in PREDICTOR_MODELS.items():
        graph_def = prepare_graph(graph_name)
        graph = CayleyGraph(graph_def)
        predictor = Predictor.pretrained(graph)
        states = torch.tensor(graph_def.central_state, device=graph.device).reshape((1, -1))
        if config.n_outputs == 1:
            assert predictor(states).shape == (1,)
        else:
            # A model with one output per generator predicts distances of all children of a state, not of the state
            # itself, so it is used through score_children.
            assert predictor.score_children(states).shape == (1, graph_def.n_generators)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="This test is slow.")
def test_lrx_14_model_finds_paths():
    # Checks that the model for "lrx-14" does what a model of this library is for: reliably finds paths in beam search.
    # With this beam width it solves random states of this graph consistently (50 out of 50 when this model was added),
    # while the Hamming distance heuristic solves none of them.
    graph = CayleyGraph(prepare_graph("lrx-14"), device="cpu", random_seed=0)
    predictor = Predictor.pretrained(graph)
    generator = torch.Generator().manual_seed(11)
    for _ in range(10):
        start_state = torch.randperm(graph.definition.state_size, generator=generator)
        result = graph.beam_search(
            start_state=start_state,
            predictor=predictor,
            beam_width=1000,
            return_path=True,
            use_child_scores=True,
        )
        assert result.path_found
        assert result.path is not None
        assert torch.equal(graph.apply_path(start_state, result.path).reshape((-1)), graph.central_state)
        # Diameter of this graph is 91, and the model is much better than the worst case: the longest of these ten
        # paths was 71 moves when this model was added.
        assert result.path_length <= 80


def test_pretrained_without_model():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    with pytest.raises(KeyError, match="No pretrained model for this graph"):
        Predictor.pretrained(graph)


def test_pretrained_rejects_model_for_another_graph(monkeypatch):
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    another_graph_def = PermutationGroups.lrx(5, k=2)
    config = ModelConfig(
        model_type="MLP",
        input_size=5,
        num_classes_for_one_hot=5,
        layers_sizes=[8],
        graph_hash=graph_hash(another_graph_def),
    )
    monkeypatch.setitem(PREDICTOR_MODELS, "lrx-5", config)
    with pytest.raises(ValueError, match="was trained for another graph"):
        Predictor.pretrained(graph)


def test_load_rejects_checkpoint_trained_for_another_graph(tmp_path):
    """Test that weights of a model trained for another graph are not loaded silently.

    A checkpoint says which graph it was trained for, so a file that happens to have weights of the right shape can
    still be detected as the wrong file - which is what keeps a `PREDICTOR_MODELS` entry from shipping nonsense.
    """
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8])
    other_graph = PermutationGroups.lrx(5, k=2)
    path = tmp_path / "weights.pt"
    save_checkpoint(path, config.build_model(), config, other_graph)

    for_this_graph = replace(config, weights_path=str(path), graph_hash=graph_hash(PermutationGroups.lrx(5)))
    with pytest.raises(ValueError, match="trained for another graph"):
        for_this_graph.load()

    # The same checkpoint loads when the config says it is for the graph the weights were trained for.
    for_that_graph = replace(config, weights_path=str(path), graph_hash=graph_hash(other_graph))
    assert for_that_graph.load() is not None
