"""Bellman (DAVI) targets - training data whose targets come from the model itself.

A random walk labels a state with the number of steps at which it visited that state, which is an upper estimate of the
true distance, and a loose one for distant states. The Bellman operator produces targets that do not have this problem,
using the fact that a state is exactly one move away from its children::

    y(s) = 0                              if s is the central state,
    y(s) = 1 + min_a V_target(child_a(s)) otherwise,

where ``V_target`` is a frozen copy of the model being trained (the `target`). Estimates then propagate outwards from
the central state, whose distance is known exactly - this is what the DAVI ("deep approximate value iteration")
algorithm does, and it is why anchors (states whose exact distance is known) must be mixed into the data: bootstrapped
targets say how far states are from each other, and something has to say where the goal is, otherwise the whole scale
drifts. :class:`BellmanTrainer` mixes anchors in and refreshes the target for you.

The recursion above looks at where the moves from a state lead, so it bootstraps the distance to the central state,
while the anchors it is mixed with come from a search starting at the central state and measure the distance from it.
These are the same distance only for inverse-closed generators, which is why :class:`BellmanTrainer` requires them - for
a graph whose generators are not inverse closed, train on the graph returned by
``CayleyGraph.with_inverted_generators``.

Example:

>>> import torch
>>> from cayleypy import CayleyGraph, PermutationGroups
>>> from cayleypy.train import bellman_targets
>>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
>>> model = lambda states: torch.full((states.shape[0],), 5.0)  # Predicts 5 for every state.
>>> # The central state, a state next to it (whose target is 1 whatever the model says), and a state further away.
>>> bellman_targets(graph, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0], [0, 2, 3, 1]]), model).targets
tensor([0., 1., 6.])
"""

import copy
import typing
from typing import Callable, Optional, Union

import torch
from torch import nn

from .config import TrainConfig
from .data import BfsAnchors, DataSource, MixtureDataSource, RandomWalksSource, TrainingData
from .trainer import Trainer
from ..predictor import Predictor

if typing.TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph
    from ..models.models import ModelConfig

# Anything that can estimate distances of states: a model, a Predictor, or any callable mapping states to scores.
TargetModel = Union[nn.Module, Predictor, Callable[[torch.Tensor], torch.Tensor]]


def _check_n_outputs(graph: "CayleyGraph", n_outputs: int) -> None:
    """Checks that a model with this number of outputs can be trained on Bellman targets for this graph."""
    n_generators = graph.definition.n_generators
    if n_outputs not in (1, n_generators):
        raise ValueError(
            f"n_outputs must be either 1 or the number of generators of this graph ({n_generators}), got {n_outputs}."
        )


def _is_central_state(graph: "CayleyGraph", encoded_states: torch.Tensor) -> torch.Tensor:
    """Returns mask of states that are the central state (compared by hash, as elsewhere in this library)."""
    return graph.hasher.make_hashes(encoded_states) == graph.central_state_hash[0]


def _values(graph: "CayleyGraph", target: Predictor, encoded_states: torch.Tensor) -> torch.Tensor:
    """Estimates distances of given states with the target model, with the distance of the central state pinned to 0.

    :param graph: Graph the states are in.
    :param target: Predictor wrapping the target model.
    :param encoded_states: States in internal representation.
    :return: Tensor of shape ``[n_states]`` with estimated distances.
    """
    with torch.no_grad():
        output = target.predict_batched(graph.decode_states(encoded_states))
    if output.dim() == 2:
        n_generators = graph.definition.n_generators
        if output.shape[1] != n_generators:
            raise ValueError(
                f"Target model returned {output.shape[1]} scores per state, but a model estimating distances of the "
                f"children of a state must return one score per generator, of which this graph has {n_generators}."
            )
        # A Q-model estimates distances of children, and a state is one move away from its nearest child.
        values = 1.0 + output.min(dim=1).values
    elif output.dim() == 1:
        values = output
    else:
        raise ValueError(
            f"Target model returned output of shape {tuple(output.shape)}, but one score per state (1-D output) or one "
            "score per generator (2-D output) was expected."
        )
    # Distances cannot be negative, and a negative estimate would drag targets of everything around it below 0. Targets
    # are detached, so that training on them never propagates gradients into the target model.
    values = values.detach().to(torch.float32).clamp_min(0.0)
    # The distance of the central state is known, whatever the model says about it - and it is the only thing that ties
    # bootstrapped targets to the actual distances.
    return torch.where(_is_central_state(graph, encoded_states), torch.zeros_like(values), values)


def bellman_targets(
    graph: "CayleyGraph",
    states: torch.Tensor,
    target: TargetModel,
    n_outputs: int = 1,
) -> TrainingData:
    """Computes Bellman targets for given states with a target model.

    The target of a state is one more than the smallest estimated distance of its children, and 0 for the central state.
    For a Q-model, whose outputs are estimated distances of the children themselves, the
    target of output ``a`` is the estimated distance of ``child_a(s)`` - so unlike targets from a random walk (see
    :class:`cayleypy.train.SparseQSampler`), all outputs are labeled.

    All estimates are clamped at 0, and the estimate for the central state is 0 regardless of what the model says. This
    makes the target of every state next to the central state exactly 1, which is what makes value iteration start.

    Computing targets needs the target model applied to all children of `states`, i.e. `n_generators` times more model
    evaluations than one training step on the same states. The target model does not have to have the same number of
    outputs as the model being trained: what is needed of it is an estimated distance of a state, and a Q-model gives
    that as one more than the smallest of its outputs.

    :param graph: Graph the states are in.
    :param states: States (in decoded representation) to compute targets for, of shape ``[n_states, state_size]``.
    :param target: Model to compute targets with - a `torch.nn.Module`, a :class:`cayleypy.Predictor`, or any callable
        mapping states to estimated distances. It is applied without gradients, but note that a module passed here is
        put in eval mode (as by :class:`cayleypy.Predictor`) and is not copied, so training it changes future targets.
    :param n_outputs: Number of outputs of the model to train - 1 for a model estimating the distance of a state, or the
        number of generators for a Q-model estimating distances of all children of a state.
    :return: States with their Bellman targets.
    """
    _check_n_outputs(graph, n_outputs)
    predictor = target if isinstance(target, Predictor) else Predictor(graph, target)
    n_generators = graph.definition.n_generators
    encoded_states = graph.encode_states(states)
    n_states = int(encoded_states.shape[0])
    child_values = _values(graph, predictor, graph.get_neighbors(encoded_states))
    # get_neighbors returns children of all states for the first generator, then for the second one, and so on.
    child_values = child_values.reshape((n_generators, n_states)).transpose(0, 1).contiguous()
    decoded_states = graph.decode_states(encoded_states)
    if n_outputs != 1:
        # Outputs of a Q-model are estimated distances of children, which is exactly what was just computed.
        return TrainingData(states=decoded_states, targets=child_values)
    targets = 1.0 + child_values.min(dim=1).values
    is_central = _is_central_state(graph, encoded_states)
    return TrainingData(states=decoded_states, targets=torch.where(is_central, torch.zeros_like(targets), targets))


class BellmanTargets(DataSource):
    """Training data whose states come from another source and whose targets come from a frozen copy of the model.

    States are taken from `states_source` (usually random walks - they are the only cheap way to reach distant states)
    and their targets are recomputed by :func:`bellman_targets`, so whatever targets that source produces are ignored.

    The target model is a frozen copy, so training the model does not change targets until :meth:`update_target` is
    called with the new weights - which is what keeps value iteration from chasing its own tail.
    :class:`BellmanTrainer` does that once every `TrainConfig.bellman_target_update_period` epochs.

    Example:

    >>> import torch
    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.train import BellmanTargets, RandomWalksSource
    >>> _ = torch.manual_seed(0)
    >>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    >>> model = torch.nn.Linear(4, 1)
    >>> walks = RandomWalksSource(graph, n_walks=4, rw_length=5)
    >>> len(BellmanTargets(graph, walks, lambda states: model(states.to(torch.float32)).squeeze(1)).generate())
    20
    """

    def __init__(
        self,
        graph: "CayleyGraph",
        states_source: DataSource,
        target_model: TargetModel,
        n_outputs: int = 1,
    ):
        """Initializes BellmanTargets.

        :param graph: Graph to compute targets in.
        :param states_source: Where to take states to label. Only states of its data are used, its targets are replaced
            by the bootstrapped ones.
        :param target_model: Model to compute targets with. It is copied and frozen, so training the original model does
            not change targets until :meth:`update_target` is called.
        :param n_outputs: Number of outputs of the model to train - 1 for a model estimating the distance of a state, or
            the number of generators for a Q-model.
        """
        _check_n_outputs(graph, n_outputs)
        self.graph = graph
        self.states_source = states_source
        self.n_outputs = int(n_outputs)
        self.target = self._frozen_copy(target_model)

    def update_target(self, model: TargetModel) -> None:
        """Replaces the target with a frozen copy of the given model, so that later targets use its weights.

        :param model: Model to compute targets with from now on.
        """
        self.target = self._frozen_copy(model)

    def generate(self) -> TrainingData:
        """Takes states from `states_source` and computes Bellman targets for them.

        :return: States with their Bellman targets.
        """
        states = self.states_source.generate().states
        return bellman_targets(self.graph, states, self.target, n_outputs=self.n_outputs)

    def _frozen_copy(self, model: TargetModel) -> Predictor:
        """Copies the given model and makes it usable for computing targets only."""
        # The graph is shared rather than copied: it is not what is being frozen here, and copying it would duplicate
        # the generators and the hasher on the device. Seeding the memo with it is what makes deepcopy share it - which
        # matters for a Predictor (and for an ensemble of them), because a Predictor holds the graph.
        target = copy.deepcopy(model, {id(self.graph): self.graph})
        inner = target.predict if isinstance(target, Predictor) else target
        if isinstance(inner, nn.Module):
            inner.eval()
            inner.requires_grad_(False)
        # A Predictor is returned as it is, so that a predictor which scores children itself keeps doing that.
        return target if isinstance(target, Predictor) else Predictor(self.graph, target)


class BellmanTrainer(Trainer):
    """Trains a model on Bellman targets, refreshing the target as training goes (the DAVI algorithm).

    This is the same training loop as :class:`cayleypy.train.Trainer`, with the data of every epoch built as:

    - states from random walks (or from `states_source`), with targets computed by :func:`bellman_targets` from a frozen
      copy of the model - see :class:`BellmanTargets`;
    - anchors - states around the central state with exact distances from a breadth-first search of depth
      `TrainConfig.bellman_anchors_depth` (1 by default, i.e. the central state and its neighbors), taking
      `TrainConfig.anchors_fraction` of the data. Unlike in :class:`cayleypy.train.Trainer`, anchors are not optional:
      bootstrapped targets only say how far states are from each other, and without exactly known distances the scale of
      the predictions drifts.

    Every `TrainConfig.bellman_target_update_period` epochs the target is refreshed from the model being trained (from
    its EMA copy, if one is maintained). Refreshing every epoch (the default) means the target is the EMA copy, i.e. it
    lags behind by design; a larger period gives a target that is held fixed for several epochs instead.

    This is normally used to fine-tune a model pretrained on random walks, whose targets are upper estimates of the
    distance - and the reason to do it is that the estimates are loose exactly where beam search spends its time.
    Fine-tuning starts from a checkpoint (:meth:`from_checkpoint`) with a learning rate lower than the one used for
    pretraining: targets move as the model moves, and a large step in that feedback loop makes training oscillate.

    Generators must be inverse closed (see the module docstring).

    Q-models (one output per generator) are trained here as well, and get more out of it than models with one output: a
    random walk labels two outputs of a state (see :class:`cayleypy.train.SparseQSampler`), while Bellman targets label
    all of them, and they do not need walks to be paths, so `TrainConfig.rw_mode` is honored for Q-models too.

    Example:

    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.models import ModelConfig
    >>> from cayleypy.train import BellmanTrainer, TrainConfig
    >>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    >>> model_config = ModelConfig(model_type="MLP", input_size=4, num_classes_for_one_hot=4, layers_sizes=[16])
    >>> train_config = TrainConfig(n_epochs=2, n_walks=8, rw_length=4, batch_size=16, lr=1e-4, seed=0)
    >>> result = BellmanTrainer(graph, model_config, train_config).train()
    >>> len(result.losses)
    2
    """

    def __init__(
        self,
        graph: "CayleyGraph",
        model_config: "ModelConfig",
        config: Optional[TrainConfig] = None,
        model: Optional[nn.Module] = None,
        states_source: Optional[DataSource] = None,
    ):
        """Initializes BellmanTrainer.

        :param graph: Graph for which to train the model.
        :param model_config: Config describing the model to train, see :class:`cayleypy.train.Trainer`.
        :param config: Configuration of the training process. Defaults to `TrainConfig()`.
        :param model: Model to train (optional). Defaults to a model built from `model_config`. Pass a model, or use
            :meth:`from_checkpoint`, to fine-tune an existing one.
        :param states_source: Where to take states to compute targets for (optional). Defaults to random walks
            described by `config`. Its targets are ignored - only states are used.
        """
        if not graph.definition.generators_inverse_closed:
            # Bellman targets are the distance to the central state, the mandatory anchors are the distance from it.
            raise ValueError(
                "BellmanTrainer requires inverse-closed generators (for every generator, its inverse must also be a "
                "generator), because its targets measure the distance to the central state while the anchors it mixes "
                "them with measure the distance from it. Train on CayleyGraph.with_inverted_generators instead."
            )
        self._states_source = states_source
        super().__init__(graph, model_config, config=config, model=model)

    def _make_data_source(self) -> DataSource:
        """Creates the mixture of Bellman targets and anchors described by the config."""
        config = self.config
        n_outputs = self.model_config.n_outputs
        states_source = self._states_source
        if states_source is None:
            states_source = RandomWalksSource(
                self.graph,
                n_walks=config.n_walks,
                rw_length=config.rw_length,
                mode=config.rw_mode,
                nbt_history_depth=config.nbt_history_depth,
            )
        # The model itself is the first target: the EMA copy does not exist yet (it is created after the data source)
        # and it starts as a copy of the model anyway.
        self.bellman_source = BellmanTargets(self.graph, states_source, self.model, n_outputs=n_outputs)
        # Anchors sample exactly their share of what the states source generates, so the mixture has nothing to throw
        # away. There are few states with exact distances near the central state, so they are sampled with repetitions.
        fraction = config.anchors_fraction
        size = max(1, int(round(fraction / (1 - fraction) * config.n_walks * config.rw_length)))
        anchors = BfsAnchors(self.graph, depth=config.bellman_anchors_depth, size=size, n_outputs=n_outputs)
        return MixtureDataSource([self.bellman_source, anchors], [1 - fraction, fraction])

    def train_epoch(self) -> float:
        """Refreshes the target if it is due, then generates data for one epoch and makes one pass over it.

        :return: Mean loss on this epoch.
        """
        if self.epoch % self.config.bellman_target_update_period == 0:
            self.bellman_source.update_target(self.model_for_inference())
        return super().train_epoch()
