import pytest
import torch

from .checkpoint import CHECKPOINT_FORMAT_VERSION, graph_hash, load_checkpoint, save_checkpoint
from .models import ModelConfig
from ..graphs_lib import MatrixGroups, PermutationGroups

MLP_CONFIG = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8, 8])
STATES = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 0, 2, 4, 3]])


def test_save_and_load(tmp_path):
    path = tmp_path / "model.pt"
    model = MLP_CONFIG.build_model()
    save_checkpoint(path, model, MLP_CONFIG)

    loaded_model, loaded_config = load_checkpoint(path)
    assert loaded_config == MLP_CONFIG
    assert not loaded_model.training  # Loaded model is in evaluation mode.
    with torch.no_grad():
        assert torch.equal(loaded_model(STATES), model(STATES))


def test_save_and_load_with_graph(tmp_path):
    path = tmp_path / "model.pt"
    graph_def = PermutationGroups.lrx(5)
    model = MLP_CONFIG.build_model()

    stored_config = save_checkpoint(path, model, MLP_CONFIG, graph_def)
    assert stored_config.graph_hash == graph_hash(graph_def)

    loaded_model, loaded_config = load_checkpoint(path, graph_def=graph_def)
    assert loaded_config.graph_hash == graph_hash(graph_def)
    with torch.no_grad():
        assert torch.equal(loaded_model(STATES), model(STATES))


def test_load_checkpoint_for_another_graph(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, MLP_CONFIG.build_model(), MLP_CONFIG, PermutationGroups.lrx(5))
    with pytest.raises(ValueError, match="was trained for another graph"):
        load_checkpoint(path, graph_def=PermutationGroups.lrx(6))


def test_load_checkpoint_without_graph_hash(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, MLP_CONFIG.build_model(), MLP_CONFIG)

    # Checkpoint has no graph hash, so there is nothing to check against the given graph.
    _, loaded_config = load_checkpoint(path, graph_def=PermutationGroups.lrx(5))
    assert loaded_config.graph_hash is None


def test_load_bare_state_dict(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(MLP_CONFIG.build_model().state_dict(), path)
    with pytest.raises(ValueError, match="not a CayleyPy checkpoint"):
        load_checkpoint(path)


def test_load_checkpoint_of_unsupported_version(tmp_path):
    path = tmp_path / "model.pt"
    model = MLP_CONFIG.build_model()
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION + 1,
        "config": MLP_CONFIG.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(path)


def test_graph_hash_is_deterministic():
    assert graph_hash(PermutationGroups.lrx(5)) == graph_hash(PermutationGroups.lrx(5))
    assert len(graph_hash(PermutationGroups.lrx(5))) == 64


def test_graph_hash_depends_on_math_only():
    graph_def = PermutationGroups.lrx(5)

    # Names of the graph and of its generators do not affect mathematical properties of the graph.
    assert graph_hash(graph_def.with_name("other name")) == graph_hash(graph_def)

    # Generators and central state do.
    assert graph_hash(PermutationGroups.lrx(6)) != graph_hash(graph_def)
    assert graph_hash(PermutationGroups.cyclic_coxeter(5)) != graph_hash(graph_def)
    assert graph_hash(graph_def.with_central_state([0, 0, 1, 1, 2])) != graph_hash(graph_def)


def test_graph_hash_for_matrix_group():
    graph_def = MatrixGroups.heisenberg()
    assert graph_hash(graph_def) == graph_hash(MatrixGroups.heisenberg())
    assert len(graph_hash(graph_def)) == 64
    assert graph_hash(MatrixGroups.heisenberg(modulo=5)) != graph_hash(graph_def)
    assert graph_hash(MatrixGroups.heisenberg(n=4)) != graph_hash(graph_def)


def test_save_and_load_for_matrix_group(tmp_path):
    path = tmp_path / "model.pt"
    graph_def = MatrixGroups.heisenberg()
    config = ModelConfig(model_type="MLP", input_size=graph_def.state_size, num_classes_for_one_hot=2, layers_sizes=[8])
    stored_config = save_checkpoint(path, config.build_model(), config, graph_def)
    assert stored_config.graph_hash == graph_hash(graph_def)

    _, loaded_config = load_checkpoint(path, graph_def=graph_def)
    assert loaded_config == stored_config
    with pytest.raises(ValueError, match="was trained for another graph"):
        load_checkpoint(path, graph_def=MatrixGroups.heisenberg(modulo=5))
