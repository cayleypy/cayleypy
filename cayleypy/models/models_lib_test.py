import os

import pytest
import torch

from .models import ModelConfig
from .models_lib import PREDICTOR_MODELS
from .. import prepare_graph, Predictor, CayleyGraph

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_loads_predictor_models():
    # Checks that all models can be loaded and successfully return prediction for central state of the graph.
    # This test does not check model quality.
    for graph_name in PREDICTOR_MODELS:
        graph_def = prepare_graph(graph_name)
        graph = CayleyGraph(graph_def)
        predictor = Predictor.pretrained(graph)
        ans = predictor(torch.tensor(graph_def.central_state).reshape((1, -1)))
        assert ans.shape == (1,)


# =============================================================================
# Offline contract tests (no Kaggle network access required)
# =============================================================================


def test_predictor_models_is_dict():
    """``PREDICTOR_MODELS`` is a dict keyed by graph name."""
    assert isinstance(PREDICTOR_MODELS, dict)
    assert len(PREDICTOR_MODELS) >= 1
    for key in PREDICTOR_MODELS:
        assert isinstance(key, str)


def test_predictor_models_have_required_fields():
    """Each ``ModelConfig`` has the required fields for loading from Kaggle."""
    required_fields = ["weights_kaggle_id", "weights_path", "model_type"]
    for graph_name, config in PREDICTOR_MODELS.items():
        assert isinstance(config, ModelConfig), f"{graph_name}: config is not ModelConfig"
        for field in required_fields:
            value = getattr(config, field, None)
            assert value is not None, f"{graph_name}: missing field {field}"
            assert isinstance(value, str), f"{graph_name}: field {field} is not a string"


def test_predictor_models_keys_match_graph_names():
    """Each key in ``PREDICTOR_MODELS`` must match a graph that ``prepare_graph`` can build."""
    for graph_name in PREDICTOR_MODELS:
        graph_def = prepare_graph(graph_name)
        assert graph_def.name == graph_name, f"Graph name mismatch for key {graph_name}"
