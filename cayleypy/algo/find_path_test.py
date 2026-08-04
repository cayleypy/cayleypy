import os
from dataclasses import replace

import pytest

from cayleypy import create_graph, PermutationGroups, CayleyGraph
from cayleypy import find_path
from cayleypy.models import ModelConfig, save_checkpoint
from cayleypy.models.models_lib import PREDICTOR_MODELS

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def test_find_path_pancake8():
    graph = CayleyGraph(PermutationGroups.pancake(8))
    start_state = [4, 7, 3, 2, 0, 5, 1, 6]
    path = find_path(graph, start_state)
    assert path is not None
    graph.validate_path(start_state, path)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
@pytest.mark.parametrize("graph_name", ["lx-9", "lrx-9", "lrx-12", "lrx-14", "lrx-15", "lrx-16", "cube_2/2/2_6gensQTM"])
def test_find_path(graph_name: str):
    graph = create_graph(name=graph_name)
    start_state = graph.random_walks(width=1, length=100)[0][-1]
    path = find_path(graph, start_state)
    assert path is not None
    graph.validate_path(start_state, path)


def test_find_path_scores_children_for_a_q_model(tmp_path, monkeypatch):
    """Test that a pretrained model with one output per generator is used through child scoring.

    Such a model estimates distances of the children of a state rather than of the state itself, so beam search must be
    called with `use_child_scores=True` - otherwise it fails outright.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    n_generators = graph.definition.n_generators
    config = ModelConfig(
        model_type="MLP",
        input_size=5,
        num_classes_for_one_hot=5,
        layers_sizes=[16],
        n_outputs=n_generators,
    )
    path_to_weights = tmp_path / "q_model.pt"
    stored_config = save_checkpoint(path_to_weights, config.build_model(), config, graph.definition)
    monkeypatch.setitem(
        PREDICTOR_MODELS, graph.definition.name, replace(stored_config, weights_path=str(path_to_weights))
    )

    options: dict = {}
    beam_search = CayleyGraph.beam_search

    def recording_beam_search(self, **kwargs):
        options.update(kwargs)
        return beam_search(self, **kwargs)

    monkeypatch.setattr(CayleyGraph, "beam_search", recording_beam_search)

    start_state = [4, 1, 0, 2, 3]
    found_path = find_path(graph, start_state)

    assert found_path is not None
    graph.validate_path(start_state, found_path)
    assert options["use_child_scores"] is True
    assert options["predictor"].n_outputs == n_generators
