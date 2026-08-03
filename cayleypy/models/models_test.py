import json
from dataclasses import replace

import kagglehub
import pytest
import torch
from kagglehub import exceptions as kagglehub_exceptions

from .checkpoint import load_checkpoint, save_checkpoint
from .models import MlpModel, ModelConfig, ResMlpModel

# Config in the format that existed before n_outputs, tokenizer_groups and graph_hash were added.
LEGACY_CONFIG_DICT = {
    "model_type": "MLP",
    "input_size": 16,
    "num_classes_for_one_hot": 16,
    "layers_sizes": [256, 256],
    "weights_kaggle_id": "fedimser/lrx-16/pyTorch/ep60/1",
    "weights_path": "model_ep60.pth",
}

# Two states of a graph with input_size=5 and num_classes_for_one_hot=5.
STATES = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])


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
    assert model(STATES).shape == (2,)


def test_build_model_unknown_type():
    config = ModelConfig(model_type="NoSuchModel", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8])
    with pytest.raises(ValueError, match="Unknown model type"):
        config.build_model()


def test_build_mlp_model_with_multiple_outputs():
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8], n_outputs=3)
    model = config.build_model()
    assert model(STATES).shape == (2, 3)


def test_mlp_state_dict_is_backward_compatible():
    # Weights of pretrained single-output models were saved before multi-output models were supported. Keys and shapes
    # of the state dict must not change, otherwise those weights stop loading.
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8])
    state_dict = config.build_model().state_dict()
    assert set(state_dict.keys()) == {
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
        "layers.3.weight",
        "layers.3.bias",
        "layers.4.weight",
        "layers.4.bias",
        "layers.6.weight",
        "layers.6.bias",
    }
    assert state_dict["layers.0.weight"].shape == (8, 25)
    assert state_dict["layers.6.weight"].shape == (1, 8)


def test_build_resmlp_model():
    config = ModelConfig(model_type="RESMLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8, 8])
    model = config.build_model()
    assert isinstance(model, ResMlpModel)
    assert model(STATES).shape == (2,)

    # There is one block per element of layers_sizes.
    assert len(model.blocks) == 3


def test_build_resmlp_model_with_multiple_outputs():
    config = ModelConfig(model_type="RESMLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8], n_outputs=3)
    model = config.build_model()
    assert model(STATES).shape == (2, 3)


def test_resmlp_has_skip_connections():
    config = ModelConfig(model_type="RESMLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8])
    model = config.build_model()

    # The first block projects the one-hot encoded state (25 features) to 8 features, so it cannot have a skip
    # connection. All subsequent blocks preserve the number of features, so they have it.
    assert not model.blocks[0].has_skip
    assert model.blocks[1].has_skip

    # With zeroed weights, a block with a skip connection is the identity function.
    block = model.blocks[1]
    torch.nn.init.zeros_(block.linear.weight)
    torch.nn.init.zeros_(block.linear.bias)
    x = torch.rand((4, 8))
    with torch.no_grad():
        assert torch.equal(block(x), x)


def test_models_are_invariant_to_batch_size():
    for model_type in ["MLP", "RESMLP"]:
        config = ModelConfig(
            model_type=model_type, input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8], n_outputs=3
        )
        model = config.build_model()
        model.eval()
        states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(6)])
        with torch.no_grad():
            expected = model(states)
            for i in range(6):
                # Batching must not change predictions (up to floating point error of matrix multiplication).
                assert torch.allclose(model(states[i : i + 1]), expected[i : i + 1], atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    for model_type in ["MLP", "RESMLP"]:
        config = ModelConfig(
            model_type=model_type, input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8], n_outputs=3
        )
        model = config.build_model()
        path = tmp_path / (model_type + ".pt")
        save_checkpoint(path, model, config)

        loaded_model, loaded_config = load_checkpoint(path)
        assert loaded_config == config
        with torch.no_grad():
            assert torch.equal(loaded_model(STATES), model(STATES))


def test_load_weights_from_checkpoint(tmp_path):
    # Weights for a model of PREDICTOR_MODELS can be saved as a self-describing checkpoint, not only as a bare
    # state dict, so that the same file can be loaded by both load_checkpoint and ModelConfig.load.
    config = ModelConfig(model_type="RESMLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8], n_outputs=3)
    model = config.build_model()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, config)
    state_dict_path = tmp_path / "state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)

    for path in [checkpoint_path, state_dict_path]:
        loaded_model = replace(config, weights_path=str(path)).load()
        with torch.no_grad():
            assert torch.equal(loaded_model(STATES), model(STATES))


def test_load_reports_failed_kaggle_download(monkeypatch):
    def fail(handle):
        raise kagglehub_exceptions.NotFoundError(f"Model {handle} not found.")

    monkeypatch.setattr(kagglehub, "model_download", fail)
    config = replace(
        ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8]),
        weights_kaggle_id="nobody/no-such-model/pyTorch/v1/1",
        weights_path="weights.pt",
    )
    with pytest.raises(RuntimeError, match="Could not download weights from Kaggle model"):
        config.load()


def test_build_model_with_non_positive_n_outputs():
    for model_type in ["MLP", "RESMLP"]:
        config = ModelConfig(
            model_type=model_type, input_size=5, num_classes_for_one_hot=5, layers_sizes=[8], n_outputs=0
        )
        with pytest.raises(ValueError, match="n_outputs must be positive"):
            config.build_model()


def test_build_resmlp_model_without_blocks():
    config = ModelConfig(model_type="RESMLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[])
    with pytest.raises(ValueError, match="at least one block"):
        config.build_model()
