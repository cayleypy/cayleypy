import pytest
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .models import ModelConfig, ResMlpModel
from .qv_model import QVModel
from ..cayley_graph import CayleyGraph
from ..graphs_lib import PermutationGroups
from ..predictor import Predictor

# Two states of a graph with input_size=5 and num_classes_for_one_hot=5.
STATES = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])


def qv_config(n_outputs: int = 3, backbone_type: str = "RESMLP", v_consistency_weight: float = 0.0) -> ModelConfig:
    return ModelConfig(
        model_type="QV",
        input_size=5,
        num_classes_for_one_hot=5,
        layers_sizes=[8, 8],
        n_outputs=n_outputs,
        backbone_type=backbone_type,
        v_consistency_weight=v_consistency_weight,
    )


def set_constant_heads(model: QVModel, q_values: list[float], v_value: float) -> None:
    """Makes the model return given values for any state, by zeroing weights of the output layer of the backbone."""
    head = model.backbone.head
    assert isinstance(head, nn.Linear)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor(q_values + [v_value]))


def test_build_qv_model():
    model = qv_config().build_model()
    assert isinstance(model, QVModel)
    assert isinstance(model.backbone, ResMlpModel)

    # The model has as many outputs as there are generators, and the backbone has one more (for the V-head).
    assert model.n_outputs == 3
    assert model.backbone.n_outputs == 4


def test_qv_model_heads_shapes():
    model = qv_config(n_outputs=4).build_model()
    q, v = model.heads(STATES)
    assert q.shape == (2, 4)
    assert v.shape == (2,)
    assert torch.equal(model.q(STATES), q)
    assert torch.equal(model.v(STATES), v)

    # Q-values are what this model returns when called, so it can be used as a Q-model.
    assert torch.equal(model(STATES), q)


def test_qv_model_with_mlp_backbone():
    model = qv_config(backbone_type="MLP").build_model()
    q, v = model.heads(STATES)
    assert q.shape == (2, 3)
    assert v.shape == (2,)


def test_qv_model_single_state():
    model = qv_config().build_model()
    q, v = model.heads(torch.tensor([0, 1, 2, 3, 4]))
    assert q.shape == (3,)
    assert v.shape == ()


def test_v_consistency_penalty_is_off_by_default():
    model = qv_config().build_model()
    assert model.v_consistency_weight == 0

    # Without the penalty, children scores are the Q-values as is.
    assert torch.equal(model.score_children(STATES), model.q(STATES))


def test_v_consistency_penalty_demotes_inconsistent_child():
    # Q says the second child is the closest one to the central state, but the V-head says this state is at distance 4,
    # so a child on an optimal path must be at distance 3 - which is what Q says about the first child only.
    model = qv_config(n_outputs=2, v_consistency_weight=2.0).build_model()
    set_constant_heads(model, [3.0, 2.0], 4.0)

    with torch.no_grad():
        assert torch.allclose(model.q(STATES), torch.tensor([[3.0, 2.0], [3.0, 2.0]]))
        assert torch.allclose(model.v(STATES), torch.tensor([4.0, 4.0]))
        scores = model.score_children(STATES)

    # Score of the first child is unchanged (its Q agrees with V-1), the second one is penalized by 2*|2-3|=2.
    assert torch.allclose(scores, torch.tensor([[3.0, 4.0], [3.0, 4.0]]))

    # The penalty changed which child looks the best.
    assert int(torch.argmin(model.q(STATES)[0])) == 1
    assert int(torch.argmin(scores[0])) == 0


def test_v_consistency_penalty_grows_with_weight():
    model = qv_config(n_outputs=2, v_consistency_weight=0.5).build_model()
    set_constant_heads(model, [3.0, 2.0], 4.0)
    with torch.no_grad():
        assert torch.allclose(model.score_children(STATES)[0], torch.tensor([3.0, 2.5]))

        # The penalty is proportional to the weight.
        model.v_consistency_weight = 1.0
        assert torch.allclose(model.score_children(STATES)[0], torch.tensor([3.0, 3.0]))


def test_qv_model_checkpoint_round_trip(tmp_path):
    graph_def = PermutationGroups.lrx(5)
    config = qv_config(n_outputs=graph_def.n_generators, v_consistency_weight=0.25)
    model = QVModel(config)
    set_constant_heads(model, [1.0, 2.0, 3.0], 4.0)
    path = tmp_path / "qv.pt"
    saved_config = save_checkpoint(path, model, config, graph_def)

    loaded_model, loaded_config = load_checkpoint(path, graph_def=graph_def)

    # Config in the checkpoint fully describes the model, including its backbone and the penalty weight.
    assert loaded_config == saved_config
    assert loaded_config.backbone_type == "RESMLP"
    assert loaded_config.v_consistency_weight == 0.25
    assert loaded_config.graph_hash is not None
    assert isinstance(loaded_model, QVModel)
    with torch.no_grad():
        assert torch.equal(loaded_model.score_children(STATES), model.score_children(STATES))


def test_qv_model_state_dict_keys():
    model = qv_config().build_model()

    # All weights of this model belong to its backbone (both heads are computed by the output layer of the backbone).
    assert all(key.startswith("backbone.") for key in model.state_dict())
    assert model.state_dict()["backbone.head.weight"].shape == (4, 8)


def test_predictor_uses_score_children_of_the_model():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    model = qv_config(n_outputs=graph_def.n_generators, v_consistency_weight=2.0).build_model()
    set_constant_heads(model, [3.0, 2.0, 2.0], 4.0)
    predictor = Predictor(graph, model)

    # The penalty is applied when Predictor scores children, so beam search sees consistency-corrected scores.
    scores = predictor.score_children(STATES)
    assert scores.shape == (2, graph_def.n_generators)
    assert torch.allclose(scores, torch.tensor([[3.0, 4.0, 4.0]] * 2))


def test_predictor_batches_score_children_of_the_model():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", batch_size=3)
    model = qv_config(n_outputs=graph_def.n_generators, v_consistency_weight=0.5).build_model()
    model.eval()
    predictor = Predictor(graph, model)
    states = torch.tensor([[i % 5, (i + 1) % 5, 2, 3, 4] for i in range(7)])

    # States are scored in several batches, which must not change the answer.
    scores = predictor.score_children(states)
    assert scores.shape == (7, graph_def.n_generators)
    with torch.no_grad():
        for i in range(7):
            assert torch.allclose(scores[i : i + 1], model.score_children(states[i : i + 1]), atol=1e-6)


def test_qv_model_without_backbone_type():
    config = ModelConfig(model_type="QV", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8], n_outputs=3)
    with pytest.raises(ValueError, match="backbone_type is not set"):
        config.build_model()


def test_qv_model_with_unknown_backbone_type():
    with pytest.raises(ValueError, match="Unknown model type: NoSuchModel"):
        qv_config(backbone_type="NoSuchModel").build_model()


def test_qv_model_with_itself_as_backbone():
    with pytest.raises(ValueError, match="cannot be its own backbone"):
        qv_config(backbone_type="QV").build_model()


def test_qv_model_with_negative_v_consistency_weight():
    with pytest.raises(ValueError, match="v_consistency_weight must be non-negative"):
        qv_config(v_consistency_weight=-0.5).build_model()


def test_qv_model_with_non_positive_n_outputs():
    with pytest.raises(ValueError, match="n_outputs must be positive"):
        qv_config(n_outputs=0).build_model()


def test_predictor_rejects_qv_model_with_wrong_number_of_outputs():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, qv_config(n_outputs=graph_def.n_generators + 1).build_model())
    with pytest.raises(ValueError, match="but the graph has 3 generators"):
        predictor.score_children(STATES)
