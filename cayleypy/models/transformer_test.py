import pytest
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .models import ModelConfig
from .transformer import TransformerModel
from ..cayley_graph import CayleyGraph
from ..predictor import Predictor
from ..puzzles import Puzzles

# Megaminx: 20 corners with 3 stickers each, then 30 edges with 2 stickers each.
MEGAMINX_GROUPS = [[3, 20], [2, 30]]

# Mini pyramorphix: 8 pieces with 3 stickers each, 17 generators.
PYRAMORPHIX_GROUPS = [[3, 8]]


def _config(**kwargs) -> ModelConfig:
    """Config of a small transformer for mini pyramorphix, with fields overridden by `kwargs`."""
    defaults = {
        "model_type": "TRANSFORMER",
        "input_size": 24,
        "num_classes_for_one_hot": 24,
        "layers_sizes": [16, 16],
        "tokenizer_groups": PYRAMORPHIX_GROUPS,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _random_states(graph_def, n_states: int) -> torch.Tensor:
    """Random states of the given graph, obtained by applying random generators to the central state."""
    generator = torch.Generator().manual_seed(42)
    state = torch.tensor(graph_def.central_state)
    states = []
    for _ in range(n_states):
        move = int(torch.randint(graph_def.n_generators, (1,), generator=generator).item())
        state = state[torch.tensor(graph_def.generators_permutations[move])]
        states.append(state)
    return torch.stack(states)


def test_output_shape_single_output():
    torch.manual_seed(0)
    model = _config().build_model()
    assert isinstance(model, TransformerModel)
    assert model.n_outputs == 1

    states = _random_states(Puzzles.mini_pyramorphix(), 5)
    assert model(states).shape == (5,)

    # A single state (1-D input) is scored to a single number, like for other models.
    assert model(states[0]).shape == ()


def test_output_shape_q_model():
    torch.manual_seed(0)
    graph_def = Puzzles.mini_pyramorphix()
    model = _config(n_outputs=graph_def.n_generators).build_model()
    assert model.n_outputs == 17

    states = _random_states(graph_def, 5)
    assert model(states).shape == (5, 17)
    assert model(states[0]).shape == (17,)


def test_megaminx_config():
    # Config from the docstring, but narrow, so the test is fast.
    torch.manual_seed(0)
    graph_def = Puzzles.megaminx()
    config = ModelConfig(
        model_type="TRANSFORMER",
        input_size=120,
        num_classes_for_one_hot=60,
        layers_sizes=[32] * 2,
        n_outputs=graph_def.n_generators,
        tokenizer_groups=MEGAMINX_GROUPS,
        n_heads=8,
        dim_feedforward=64,
    )
    model = config.build_model()
    assert model.tokenizer.n_tokens == 50
    assert model.token_embedding.num_embeddings == 60  # Vocabulary is determined by the tokenizer, not by the config.
    assert model.token_type_embedding.num_embeddings == 2  # Corners and edges.
    assert model.position_embedding.shape == (50, 32)
    assert len(model.encoder.layers) == 2

    states = _random_states(graph_def, 3)
    assert model(states).shape == (3, 24)


def test_defaults_for_heads_and_feedforward():
    torch.manual_seed(0)
    model = _config(layers_sizes=[128, 128, 128]).build_model()
    assert len(model.encoder.layers) == 3
    # One head per 64 features, feed-forward layer 4 times wider than the model.
    assert model.encoder.layers[0].self_attn.num_heads == 2
    assert model.encoder.layers[0].linear1.out_features == 512

    # For a narrow model, the default is a single attention head.
    assert _config(layers_sizes=[16]).build_model().encoder.layers[0].self_attn.num_heads == 1


def test_output_does_not_depend_on_batch_size():
    torch.manual_seed(0)
    model = _config(n_outputs=17).build_model()
    model.eval()
    states = _random_states(Puzzles.mini_pyramorphix(), 6)

    scores = model(states)
    assert torch.allclose(scores[:2], model(states[:2]), atol=1e-6)
    for i in range(6):
        assert torch.allclose(scores[i], model(states[i]), atol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    graph_def = Puzzles.mini_pyramorphix()
    config = _config(n_outputs=graph_def.n_generators, n_heads=4, dim_feedforward=48)
    model = config.build_model()
    states = _random_states(graph_def, 4)
    expected = model(states)

    path = tmp_path / "transformer.pt"
    save_checkpoint(path, model, config, graph_def)
    loaded_model, loaded_config = load_checkpoint(path, graph_def=graph_def)

    assert isinstance(loaded_model, TransformerModel)
    assert loaded_config.n_heads == 4
    assert loaded_config.dim_feedforward == 48
    assert loaded_config.tokenizer_groups == PYRAMORPHIX_GROUPS
    assert torch.allclose(loaded_model(states), expected, atol=1e-6)


def test_works_as_predictor():
    torch.manual_seed(0)
    graph = CayleyGraph(Puzzles.mini_pyramorphix(), device="cpu")
    predictor = Predictor(graph, _config().build_model())
    states = _random_states(graph.definition, 3)

    assert predictor(states).shape == (3,)
    # Default implementation of score_children applies this single-output model to all children of every state.
    assert predictor.score_children(states).shape == (3, 17)


def test_without_tokenizer_groups():
    with pytest.raises(ValueError, match="tokenizer_groups"):
        _config(tokenizer_groups=None).build_model()


def test_tokenizer_groups_inconsistent_with_input_size():
    with pytest.raises(ValueError, match="input_size in the config is 25"):
        _config(input_size=25).build_model()


def test_no_layers():
    with pytest.raises(ValueError, match="layers_sizes is empty"):
        _config(layers_sizes=[]).build_model()


def test_layers_of_different_width():
    with pytest.raises(ValueError, match="All encoder layers must have the same width"):
        _config(layers_sizes=[16, 32]).build_model()


def test_width_not_divisible_by_number_of_heads():
    with pytest.raises(ValueError, match="must be divisible by the number of attention heads"):
        _config(layers_sizes=[16, 16], n_heads=3).build_model()


def test_invalid_n_heads():
    with pytest.raises(ValueError, match="n_heads must be positive"):
        _config(n_heads=0).build_model()


def test_invalid_dim_feedforward():
    with pytest.raises(ValueError, match="dim_feedforward must be positive"):
        _config(dim_feedforward=0).build_model()


def test_invalid_n_outputs():
    with pytest.raises(ValueError, match="n_outputs must be positive"):
        _config(n_outputs=0).build_model()
