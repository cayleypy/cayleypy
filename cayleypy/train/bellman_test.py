import os

import pytest
import torch

from .bellman import BellmanTargets, BellmanTrainer, bellman_targets
from .config import TrainConfig
from .data import BfsAnchors, DataSource, MixtureDataSource, RandomWalksSource, TrainingData
from .trainer import Trainer
from ..cayley_graph import CayleyGraph
from ..graphs_lib import PermutationGroups
from ..models.models import ModelConfig
from ..predictor import Predictor

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"

MLP_CONFIG = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[16, 16])

# Config of a Q-model for lrx(5) - one output per generator. Architectures with several outputs are added in another
# change, so tests below train the model defined here instead of building one from this config.
Q_CONFIG = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[16], n_outputs=3)

# Configuration making training as short as possible - for tests that check mechanics rather than model quality.
TINY_CONFIG = TrainConfig(n_epochs=2, n_walks=4, rw_length=3, batch_size=8, seed=0)


def _lrx5() -> CayleyGraph:
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu")


def _all_states_with_distances(graph: CayleyGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns all states of the graph with their true distances, computed by exact BFS."""
    layers = graph.bfs(max_layer_size_to_store=None).layers
    states = torch.vstack([torch.as_tensor(layer) for layer in layers.values()])
    distances = torch.hstack([torch.full((len(layer),), float(distance)) for distance, layer in layers.items()])
    return states, distances


def _keys(states: torch.Tensor) -> torch.Tensor:
    """Maps states of a permutation group to unique integers, so that they can be looked up in a table."""
    base = states.shape[1]
    powers = base ** torch.arange(base, dtype=torch.int64)
    return (states.to(torch.int64) * powers).sum(dim=1)


class _TableModel(torch.nn.Module):
    """Model that looks up the value of a state in a table, e.g. an oracle knowing exact distances."""

    def __init__(self, states: torch.Tensor, values: torch.Tensor):
        super().__init__()
        keys = _keys(states)
        order = torch.argsort(keys)
        self.keys = keys[order]
        self.values = values[order]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        keys = _keys(states)
        index = torch.searchsorted(self.keys, keys)
        assert bool((self.keys[index] == keys).all()), "Table has no value for some of these states."
        return self.values[index]


class _ConstantModel(torch.nn.Module):
    """Model predicting the same value for every state, with a weight so that training can change it."""

    def __init__(self, value: float, n_outputs: int = 1):
        super().__init__()
        self.n_outputs = n_outputs
        self.value = torch.nn.Parameter(torch.full((n_outputs,), float(value)))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        answer = self.value.expand(states.shape[0], self.n_outputs)
        return answer if self.n_outputs > 1 else answer.reshape(-1)


class _QModel(torch.nn.Module):
    """Minimal model with one output per generator, standing in for a Q-model architecture."""

    def __init__(self, state_size: int = 5, n_outputs: int = 3):
        super().__init__()
        self.n_outputs = n_outputs
        self.layer = torch.nn.Linear(state_size, n_outputs)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layer(states.to(torch.float32))


class _FixedStates(DataSource):
    """Data source always returning the same states (their targets are ignored by BellmanTargets)."""

    def __init__(self, states: torch.Tensor):
        self.states = states

    def generate(self) -> TrainingData:
        return TrainingData(states=self.states, targets=torch.zeros((self.states.shape[0],)))


def _exact_q_values(graph: CayleyGraph, states: torch.Tensor, oracle: _TableModel) -> torch.Tensor:
    """Returns exact distances of all children of given states, of shape ``[n_states, n_generators]``."""
    columns = [oracle(graph.apply_path(states, [i])) for i in range(graph.definition.n_generators)]
    return torch.stack(columns, dim=1)


def test_bellman_targets_are_exact_distances_when_the_target_model_is_exact():
    # The Bellman operator has exact distances as a fixed point: d(s) = 1 + min over children of d.
    graph = _lrx5()
    states, distances = _all_states_with_distances(graph)
    data = bellman_targets(graph, states, _TableModel(states, distances))
    assert torch.equal(data.targets, distances)
    assert data.mask is None and data.weights is None


def test_bellman_targets_for_q_model_are_exact_distances_of_children():
    graph = _lrx5()
    states, distances = _all_states_with_distances(graph)
    oracle = _TableModel(states, distances)
    q_values = _exact_q_values(graph, states, oracle)

    # A Q-model knowing exact distances of children is a fixed point too.
    data = bellman_targets(graph, states, _TableModel(states, q_values), n_outputs=3)
    assert data.targets.shape == (120, 3)
    assert torch.equal(data.targets, q_values)
    # Column by column, because comparing whole tensors would not notice generators being transposed.
    for gen_id in range(3):
        assert torch.equal(data.targets[:, gen_id], oracle(graph.apply_path(states, [gen_id])))


def test_bellman_targets_for_q_model_can_come_from_a_model_with_one_output():
    # Targets of a Q-model are estimated distances of children, and any model can estimate those.
    graph = _lrx5()
    states, distances = _all_states_with_distances(graph)
    oracle = _TableModel(states, distances)
    data = bellman_targets(graph, states, oracle, n_outputs=3)
    assert torch.equal(data.targets, _exact_q_values(graph, states, oracle))


def test_bellman_targets_pin_the_central_state_and_its_neighbors():
    graph = _lrx5()
    central = graph.central_state.reshape(1, -1)
    neighbor = graph.apply_path(central, [2])
    far_state = graph.apply_path(central, [2, 0])
    states = torch.vstack([central, neighbor, far_state])

    targets = bellman_targets(graph, states, _ConstantModel(100.0)).targets
    # The distance of the central state is known, so it is 0 and the state next to it is 1, whatever the model says.
    assert float(targets[0]) == 0.0
    assert float(targets[1]) == 1.0
    assert float(targets[2]) == 101.0


def test_bellman_targets_clamp_negative_estimates():
    graph = _lrx5()
    states, _ = _all_states_with_distances(graph)
    targets = bellman_targets(graph, states, _ConstantModel(-7.0)).targets
    # Distances cannot be negative, so estimates are clamped at 0 and every state is "one move from something at 0".
    assert torch.equal(targets, torch.where(_keys(states) == _keys(graph.central_state.reshape(1, -1)), 0.0, 1.0))


def test_bellman_targets_use_the_nearest_child():
    graph = _lrx5()
    # A state 3 moves away, so that none of its children is the central state (whose estimate is pinned to 0).
    state = graph.apply_path(graph.central_state.reshape(1, -1), [2, 0, 2])
    children = graph.get_neighbors_decoded(state)
    model = _TableModel(children, torch.tensor([7.0, 2.0, 5.0]))
    assert float(bellman_targets(graph, state, model).targets[0]) == 3.0
    # For a Q-model, every output gets the estimate for its own child instead.
    assert torch.equal(bellman_targets(graph, state, model, n_outputs=3).targets, torch.tensor([[7.0, 2.0, 5.0]]))


def test_bellman_targets_accept_a_predictor():
    graph = _lrx5()
    states, _ = _all_states_with_distances(graph)
    from_predictor = bellman_targets(graph, states, Predictor(graph, "hamming")).targets
    from_callable = bellman_targets(graph, states, lambda x: (x != graph.central_state).sum(dim=1)).targets
    assert torch.equal(from_predictor, from_callable)
    assert torch.all(from_predictor >= 0)


def test_bellman_targets_reject_target_scoring_a_wrong_number_of_children():
    graph = _lrx5()
    # Two scores per state, while a model estimating distances of children of this graph must return three.
    target = _ConstantModel(1.0, n_outputs=2)
    with pytest.raises(ValueError, match="one score per generator"):
        bellman_targets(graph, graph.central_state.reshape(1, -1), target)


def test_target_given_as_predictor_is_frozen_and_does_not_copy_the_graph():
    graph = _lrx5()
    model = _QModel()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    source = BellmanTargets(graph, _FixedStates(graph.central_state.reshape(1, -1)), Predictor(graph, model), 3)

    assert source.target.graph is graph
    frozen = source.target.predict
    assert isinstance(frozen, torch.nn.Module)
    assert frozen is not model
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_bellman_targets_split_large_input_into_batches():
    # Children of these states do not fit in one batch, so the target model is applied several times.
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", batch_size=7)
    states, distances = _all_states_with_distances(graph)
    assert torch.equal(bellman_targets(graph, states, _TableModel(states, distances)).targets, distances)


def test_bellman_targets_accept_a_single_state():
    graph = _lrx5()
    data = bellman_targets(graph, graph.central_state, _ConstantModel(3.0))
    assert data.states.shape == (1, 5)
    assert torch.equal(data.targets, torch.zeros(1))


def test_value_iteration_converges_to_exact_distances():
    # Applying the operator to its own output propagates exact distances outwards from the central state, one layer of
    # the graph per iteration - which is what training on Bellman targets does, with a model in between.
    graph = _lrx5()
    states, distances = _all_states_with_distances(graph)
    values = torch.zeros(int(states.shape[0]))
    diameter = int(distances.max())
    for iteration in range(diameter):
        values = bellman_targets(graph, states, _TableModel(states, values)).targets
        # After k iterations, states at distance at most k have their exact distance, and the rest are underestimated.
        assert torch.equal(values, distances.clamp(max=iteration + 1))
    assert torch.equal(values, distances)


def test_bellman_targets_source_labels_states_of_the_inner_source():
    graph = _lrx5()
    torch.manual_seed(0)
    walks = RandomWalksSource(graph, n_walks=4, rw_length=5)
    source = BellmanTargets(graph, walks, _ConstantModel(2.0))
    data = source.generate()
    assert len(data) == 20
    assert data.targets.shape == (20,)
    # Targets are recomputed from the model, whatever targets the inner source produced.
    assert torch.equal(data.targets, bellman_targets(graph, data.states, _ConstantModel(2.0)).targets)


def test_bellman_targets_source_freezes_the_target_model():
    graph = _lrx5()
    states, _ = _all_states_with_distances(graph)
    model = _ConstantModel(2.0)
    source = BellmanTargets(graph, _FixedStates(states), model)
    before = source.generate().targets

    with torch.no_grad():
        model.value += 10.0
    # Training the model does not change targets: the source holds a frozen copy of it.
    assert torch.equal(source.generate().targets, before)

    source.update_target(model)
    after = source.generate().targets
    assert not torch.equal(after, before)
    assert torch.equal(after, bellman_targets(graph, states, model).targets)


def test_bellman_targets_source_for_q_model():
    graph = _lrx5()
    states, _ = _all_states_with_distances(graph)
    source = BellmanTargets(graph, _FixedStates(states), _QModel(), n_outputs=3)
    data = source.generate()
    assert data.targets.shape == (120, 3)
    # Unlike targets from a random walk, all outputs are labeled.
    assert data.mask is None


def test_bellman_trainer_mixes_bellman_targets_with_anchors():
    graph = _lrx5()
    config = TrainConfig(n_epochs=1, n_walks=4, rw_length=5, batch_size=8, anchors_fraction=0.5, seed=0)
    trainer = BellmanTrainer(graph, MLP_CONFIG, config)
    source = trainer.data_source
    assert isinstance(source, MixtureDataSource)
    assert isinstance(source.sources[0], BellmanTargets)
    assert isinstance(source.sources[1], BfsAnchors)
    assert source.fractions == [0.5, 0.5]
    assert source.sources[0] is trainer.bellman_source


def test_bellman_trainer_anchors_have_exact_targets():
    graph = _lrx5()
    config = TrainConfig(n_epochs=1, n_walks=8, rw_length=5, anchors_fraction=0.5, bellman_anchors_depth=2, seed=0)
    trainer = BellmanTrainer(graph, MLP_CONFIG, config, model=_ConstantModel(100.0))
    data = trainer.generate_data()
    # Walks give 8*5 states, and anchors are sampled to the same number, because they are half of the data.
    assert len(data) == 80
    # With a model predicting 100 everywhere, Bellman targets can only be 0, 1 or 101, so a target of 2 is a state
    # whose distance the breadth-first search knows exactly.
    assert int((data.targets == 2.0).sum()) > 0
    n_exact = int(((data.targets != 0.0) & (data.targets != 1.0) & (data.targets != 101.0)).sum())
    assert n_exact > 0.2 * len(data)


def test_bellman_trainer_refreshes_the_target_every_epoch():
    graph = _lrx5()
    config = TrainConfig(n_epochs=3, n_walks=4, rw_length=5, batch_size=8, lr=0.05, ema_decay=0, seed=0)
    trainer = BellmanTrainer(graph, MLP_CONFIG, config)
    states, _ = _all_states_with_distances(graph)
    for _ in range(config.n_epochs):
        with torch.no_grad():
            expected = trainer.predictor()(states).clone()
        trainer.train_epoch()
        with torch.no_grad():
            # The target is the model as it was at the beginning of the epoch.
            assert torch.equal(trainer.bellman_source.target(states), expected)
            assert not torch.equal(trainer.predictor()(states), expected)


def test_bellman_trainer_keeps_the_target_for_the_configured_number_of_epochs():
    graph = _lrx5()
    config = TrainConfig(
        n_epochs=4, n_walks=4, rw_length=5, batch_size=8, lr=0.05, ema_decay=0, bellman_target_update_period=2, seed=0
    )
    trainer = BellmanTrainer(graph, MLP_CONFIG, config)
    states, _ = _all_states_with_distances(graph)

    trainer.train_epoch()
    with torch.no_grad():
        after_first_refresh = trainer.bellman_source.target(states).clone()
    trainer.train_epoch()
    with torch.no_grad():
        # Second epoch does not refresh the target, so it is still the same.
        assert torch.equal(trainer.bellman_source.target(states), after_first_refresh)
        expected = trainer.predictor()(states).clone()
    trainer.train_epoch()
    with torch.no_grad():
        # Third epoch refreshes it, to the weights the model had when that epoch started.
        assert torch.equal(trainer.bellman_source.target(states), expected)


def test_training_on_bellman_targets_reduces_the_loss():
    graph = _lrx5()
    trainer = BellmanTrainer(graph, MLP_CONFIG, TrainConfig(n_walks=8, rw_length=5, lr=0.01, seed=0))
    data = trainer.generate_data()
    first_loss = trainer.train_step(data.states, data.targets)
    for _ in range(10):
        last_loss = trainer.train_step(data.states, data.targets)
    assert last_loss < first_loss


def test_bellman_trainer_trains_a_q_model():
    graph = _lrx5()
    config = TrainConfig(n_epochs=2, n_walks=8, rw_length=5, batch_size=16, lr=0.01, seed=0)
    trainer = BellmanTrainer(graph, Q_CONFIG, config, model=_QModel())
    data = trainer.generate_data()
    assert data.targets.shape[1] == 3
    assert data.mask is None
    result = trainer.train()
    assert len(result.losses) == 2
    with torch.no_grad():
        assert trainer.predictor().predict_batched(data.states).shape == (len(data), 3)


def test_bellman_trainer_from_checkpoint_continues_from_the_pretrained_model(tmp_path):
    path = tmp_path / "pretrained.pt"
    graph = _lrx5()
    pretrained = Trainer(graph, MLP_CONFIG, TINY_CONFIG)
    pretrained.train()
    pretrained.save(path)

    # Fine-tuning normally uses a lower learning rate than pretraining, because targets move as the model moves.
    trainer = BellmanTrainer.from_checkpoint(path, graph, TrainConfig(n_epochs=1, n_walks=4, rw_length=3, lr=1e-4))
    assert isinstance(trainer, BellmanTrainer)
    states, _ = _all_states_with_distances(graph)
    with torch.no_grad():
        assert torch.equal(trainer.model(states), pretrained.ema_model(states))
    trainer.train()
    with torch.no_grad():
        assert not torch.equal(trainer.model(states), pretrained.ema_model(states))


def test_bellman_trainer_does_not_move_the_scale_without_optimization():
    # Sanity check: bootstrapped targets by themselves change nothing - predictions only move when weights do.
    graph = _lrx5()
    config = TrainConfig(n_epochs=5, n_walks=8, rw_length=5, batch_size=16, lr=1e-12, seed=0)
    trainer = BellmanTrainer(graph, MLP_CONFIG, config)
    states, _ = _all_states_with_distances(graph)
    with torch.no_grad():
        before = trainer.predictor()(states).clone()
    trainer.train()
    with torch.no_grad():
        assert torch.allclose(trainer.predictor()(states), before, atol=1e-5)


def test_bellman_targets_reject_wrong_number_of_outputs():
    graph = _lrx5()
    states = graph.central_state.reshape(1, -1)
    for n_outputs in (0, 2, 4):
        with pytest.raises(ValueError, match="n_outputs must be either 1 or the number of generators"):
            bellman_targets(graph, states, _ConstantModel(1.0), n_outputs=n_outputs)
    with pytest.raises(ValueError, match="n_outputs must be either 1 or the number of generators"):
        BellmanTargets(graph, _FixedStates(states), _ConstantModel(1.0), n_outputs=2)


def test_bellman_targets_reject_target_model_with_unexpected_output_shape():
    graph = _lrx5()
    states = graph.central_state.reshape(1, -1)

    def model(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 3, 2))

    with pytest.raises(ValueError, match="one score per state .1-D output. or one score per generator"):
        bellman_targets(graph, states, model)


def test_train_config_rejects_invalid_bellman_values():
    with pytest.raises(ValueError, match="bellman_anchors_depth must be at least 1"):
        TrainConfig(bellman_anchors_depth=0)
    with pytest.raises(ValueError, match="bellman_anchors_depth must be at least 1"):
        TrainConfig(bellman_anchors_depth=-1)
    with pytest.raises(ValueError, match="bellman_target_update_period must be positive"):
        TrainConfig(bellman_target_update_period=0)


def test_bellman_trainer_rejects_checkpoint_for_another_graph(tmp_path):
    path = tmp_path / "model.pt"
    Trainer(_lrx5(), MLP_CONFIG, TINY_CONFIG).save(path)
    another_graph = CayleyGraph(PermutationGroups.lrx(5, k=2), device="cpu")
    with pytest.raises(ValueError, match="was trained for another graph"):
        BellmanTrainer.from_checkpoint(path, another_graph, TINY_CONFIG)


def test_bellman_trainer_rejects_model_with_wrong_number_of_outputs():
    config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[16], n_outputs=4)
    with pytest.raises(ValueError, match="one output per generator of this graph, of which there are 3"):
        BellmanTrainer(_lrx5(), config, TINY_CONFIG, model=_QModel(n_outputs=4))


def _mean_absolute_error(graph: CayleyGraph, predictor: Predictor) -> float:
    """Mean absolute error of the predictor on all states of the graph, against exact distances."""
    states, distances = _all_states_with_distances(graph)
    with torch.no_grad():
        return float((predictor(states) - distances).abs().mean())


def _prediction_at_central_state(graph: CayleyGraph, predictor: Predictor) -> float:
    with torch.no_grad():
        return float(predictor(graph.central_state.reshape(1, -1))[0])


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="Slow test.")
def test_bellman_finetuning_improves_a_pretrained_model(tmp_path):
    # Random walks label a state with the number of steps at which they visited it, which overestimates its distance.
    # Bellman targets do not have this bias, so fine-tuning on them moves predictions towards the exact distances.
    graph = _lrx5()
    path = tmp_path / "pretrained.pt"
    pretrain_config = TrainConfig(n_epochs=100, n_walks=64, rw_length=12, batch_size=128, lr=0.01, seed=42)
    pretrained = Trainer(graph, MLP_CONFIG, pretrain_config)
    pretrained.train()
    pretrained.save(path)
    error_before = _mean_absolute_error(graph, pretrained.predictor())
    # Walks even overestimate the distance of the central state, which they visit at every step of every walk.
    assert abs(_prediction_at_central_state(graph, pretrained.predictor())) > 1.0

    finetune_config = TrainConfig(
        n_epochs=100, n_walks=64, rw_length=12, batch_size=128, lr=0.001, bellman_anchors_depth=2, seed=42
    )
    finetuned = BellmanTrainer.from_checkpoint(path, graph, finetune_config)
    finetuned.train()
    error_after = _mean_absolute_error(graph, finetuned.predictor())
    assert error_after < 0.7 * error_before
    assert abs(_prediction_at_central_state(graph, finetuned.predictor())) < 0.5


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="Slow test.")
def test_bellman_training_from_scratch_learns_exact_distances():
    # Value iteration works without any pretraining at all: exact distances spread from the central state, and on a
    # graph small enough for the model to fit all of it, predictions end up close to the distances from exact BFS.
    graph = _lrx5()
    model_config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[64, 64])
    config = TrainConfig(
        n_epochs=200, n_walks=64, rw_length=12, batch_size=128, lr=0.01, bellman_anchors_depth=2, seed=0
    )
    trainer = BellmanTrainer(graph, model_config, config)
    trainer.train()
    # Predicting the mean distance for every state would give an error of 1.47 on this graph.
    assert _mean_absolute_error(graph, trainer.predictor()) < 0.5
    assert abs(_prediction_at_central_state(graph, trainer.predictor())) < 0.3


def test_bellman_trainer_honors_the_walk_mode_for_a_q_model():
    """Test that walks are generated in the configured mode for a Q-model too (unlike in `Trainer`)."""
    graph = _lrx5()
    config = TrainConfig(n_epochs=1, n_walks=4, rw_length=3, batch_size=8, rw_mode="bfs", seed=0)

    trainer = BellmanTrainer(graph, Q_CONFIG, config, model=_QModel())

    states_source = trainer.bellman_source.states_source
    assert isinstance(states_source, RandomWalksSource)
    assert states_source.mode == "bfs"


def test_bellman_trainer_rejects_graph_without_inverse_closed_generators():
    """Test that a graph whose distances go one way only is rejected, instead of training on mixed directions."""
    graph = CayleyGraph(PermutationGroups.lx(5), device="cpu")
    model_config = ModelConfig(model_type="MLP", input_size=5, num_classes_for_one_hot=5, layers_sizes=[8])

    with pytest.raises(ValueError, match="inverse-closed generators"):
        BellmanTrainer(graph, model_config, TrainConfig(n_epochs=1, n_walks=4, rw_length=3))
