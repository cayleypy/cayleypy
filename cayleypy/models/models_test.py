import json

import pytest
import torch

from .models import MlpModel, ModelConfig

# Config in the format that existed before n_outputs, tokenizer_groups and graph_hash were added.
LEGACY_CONFIG_DICT = {
    "model_type": "MLP",
    "input_size": 16,
    "num_classes_for_one_hot": 16,
    "layers_sizes": [256, 256],
    "weights_kaggle_id": "fedimser/lrx-16/pyTorch/ep60/1",
    "weights_path": "model_ep60.pth",
}


def test_from_dict_legacy():
    config = ModelConfig.from_dict(LEGACY_CONFIG_DICT)
    assert config.model_type == "MLP"
    assert config.input_size == 16
    assert config.num_classes_for_one_hot == 16
    assert config.layers_sizes == [256, 256]
    assert config.weights_kaggle_id == "fedimser/lrx-16/pyTorch/ep60/1"
    assert config.weights_path == "model_ep60.pth"

    # Defaults describe single-output model without tokenization, not tied to any graph.
    assert config.n_outputs == 1
    assert config.tokenizer_groups is None
    assert config.graph_hash is None


def test_from_dict_new_fields():
    config = ModelConfig.from_dict(
        {
            "model_type": "MLP",
            "input_size": 120,
            "num_classes_for_one_hot": 60,
            "layers_sizes": [64],
            "n_outputs": 12,
            "tokenizer_groups": [[3, 20], [2, 30]],
            "graph_hash": "abc123",
        }
    )
    assert config.n_outputs == 12
    assert config.tokenizer_groups == [[3, 20], [2, 30]]
    assert config.graph_hash == "abc123"


def test_to_dict_round_trip():
    config = ModelConfig(
        model_type="MLP",
        input_size=120,
        num_classes_for_one_hot=60,
        layers_sizes=[64, 32],
        n_outputs=12,
        tokenizer_groups=[[3, 20], [2, 30]],
        graph_hash="abc123",
    )
    as_dict = config.to_dict()

    # Config must be convertible to a dict of primitives (so it can be stored in a checkpoint).
    assert as_dict["layers_sizes"] == [64, 32]
    assert as_dict["tokenizer_groups"] == [[3, 20], [2, 30]]
    assert ModelConfig.from_dict(as_dict) == config


def test_to_dict_is_json_serializable():
    config = ModelConfig.from_dict(LEGACY_CONFIG_DICT)
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()


def test_build_mlp_model():
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8])
    model = config.build_model()
    assert isinstance(model, MlpModel)
    states = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    assert model(states).shape == (2,)


def test_build_model_unknown_type():
    config = ModelConfig(model_type="NoSuchModel", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8])
    with pytest.raises(ValueError, match="Unknown model type"):
        config.build_model()


def test_build_mlp_model_with_multiple_outputs():
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8], n_outputs=3)
    with pytest.raises(ValueError, match="n_outputs"):
        config.build_model()
