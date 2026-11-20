"""Trainer for models estimating distances in a Cayley graph."""

import copy
import math
import typing
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import TrainConfig
from ..models.checkpoint import PathType, load_checkpoint, save_checkpoint
from ..models.models import ModelConfig
from ..predictor import Predictor

if typing.TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


@dataclass
class TrainResult:
    """Result of training.

    :param losses: Mean loss on every epoch. Because data is regenerated on every epoch, these are losses on states
        the model has not seen before, so they can be read as a validation curve.
    :param n_steps: Total number of optimizer steps made.
    """

    losses: list[float]
    n_steps: int


def _validate_model_config(model_config: ModelConfig, graph: "CayleyGraph") -> None:
    """Checks that a model described by this config can be trained for this graph."""
    state_size = graph.definition.state_size
    if model_config.input_size != state_size:
        raise ValueError(
            f"Model expects states of size {model_config.input_size}, but states of this graph have size {state_size}."
        )
    if model_config.n_outputs != 1:
        raise ValueError(
            f"Trainer supports only models with a single output, got n_outputs={model_config.n_outputs}. A model with "
            "one output per generator needs a target for every output, which random walks do not provide."
        )
    # Models consuming tokenized states do not use one-hot encoding, so this only applies to the other ones.
    if model_config.tokenizer_groups is None:
        max_value = int(graph.central_state.max())
        if model_config.num_classes_for_one_hot <= max_value:
            raise ValueError(
                f"Model one-hot encodes states with {model_config.num_classes_for_one_hot} classes, which is not "
                f"enough for this graph, where an element of a state can be as large as {max_value}."
            )


def _warn_if_not_inverse_closed(graph: "CayleyGraph") -> None:
    """Warns that targets generated for this graph need not measure the distance a predictor is asked about."""
    if graph.definition.generators_inverse_closed:
        return
    warnings.warn(
        "Generators of this graph are not inverse closed, so the random walks used for training measure the distance "
        "from the central state, which need not be the distance to it - and a predictor is asked for the latter. To "
        "get targets in the direction a predictor is asked about, train on CayleyGraph.with_inverted_generators. This "
        "check looks at the generators alone, so it is a false alarm for a graph whose induced action on states is "
        "symmetric anyway.",
        stacklevel=3,
    )


class Trainer:
    """Trains a model to estimate distance from the central state of a Cayley graph.

    On every epoch, a fresh set of random walks is generated, and the number of steps at which a state was visited is
    used as its target distance. These targets are upper estimates of the true distance (see
    :class:`cayleypy.algo.RandomWalksGenerator`), which is why fast-mixing walks ("nbt") are the default.

    The model is fitted with AdamW, the learning rate follows a cosine schedule, and an exponential moving average
    (EMA) of the weights is maintained - it is usually a better predictor than the weights themselves, and it is what
    :meth:`predictor` and :meth:`save` use by default.

    Training progress is reported with `print` when `TrainConfig.verbose` is at least 1.

    Walks start at the central state, so what their targets measure is the distance from it, while a predictor is asked
    for the distance to it. These are the same distance when the generators of the graph are inverse closed, and the
    trainer warns when they are not.

    Example:

    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.models import ModelConfig
    >>> from cayleypy.train import TrainConfig, Trainer
    >>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    >>> model_config = ModelConfig(model_type="MLP", input_size=4, num_classes_for_one_hot=4, layers_sizes=[16])
    >>> train_config = TrainConfig(n_epochs=2, n_walks=8, rw_length=4, batch_size=16, seed=0)
    >>> result = Trainer(graph, model_config, train_config).train()
    >>> len(result.losses)
    2
    """

    def __init__(
        self,
        graph: "CayleyGraph",
        model_config: ModelConfig,
        config: Optional[TrainConfig] = None,
        model: Optional[nn.Module] = None,
    ):
        """Initializes Trainer.

        :param graph: Graph for which to train the model.
        :param model_config: Config describing the model to train. It is also stored in checkpoints written by
            :meth:`save`, which makes them self-describing.
        :param config: Configuration of the training process. Defaults to `TrainConfig()`.
        :param model: Model to train (optional). Defaults to a model built from `model_config` with randomly
            initialized weights. Pass a model to continue training an existing one - it must be described by
            `model_config`.
        """
        _validate_model_config(model_config, graph)
        _warn_if_not_inverse_closed(graph)
        self.graph = graph
        self.model_config = model_config
        self.config = config if config is not None else TrainConfig()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.model = (model_config.build_model() if model is None else model).to(graph.device)
        self.loss = self.config.make_loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.n_epochs, eta_min=self.config.lr_min
        )
        self.ema_model: Optional[nn.Module] = None
        if self.config.ema_decay > 0:
            self.ema_model = copy.deepcopy(self.model).eval()
            for parameter in self.ema_model.parameters():
                parameter.requires_grad_(False)
        self.epoch = 0
        self.n_steps = 0

    @staticmethod
    def from_checkpoint(
        path: PathType,
        graph: "CayleyGraph",
        config: Optional[TrainConfig] = None,
    ) -> "Trainer":
        """Creates trainer continuing training of a model stored in a checkpoint.

        Only weights are stored in checkpoints, so the optimizer, the learning rate schedule and the EMA copy start
        anew (the EMA copy starts from the weights that were loaded).

        :param path: Path to a checkpoint written by :func:`cayleypy.models.save_checkpoint` (e.g. by :meth:`save`).
        :param graph: Graph for which to continue training. If the checkpoint says which graph the model was trained
            for, it must be this graph.
        :param config: Configuration of the training process. Defaults to `TrainConfig()`.
        :return: The trainer.
        """
        model, model_config = load_checkpoint(path, device=str(graph.device), graph_def=graph.definition)
        return Trainer(graph, model_config, config=config, model=model)

    @property
    def learning_rate(self) -> float:
        """Learning rate that will be used on the next optimizer step."""
        return float(self.optimizer.param_groups[0]["lr"])

    def generate_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates training data for one epoch - random walks with estimated distances.

        :return: Pair of tensors (states, targets), where ``targets[i]`` is the estimated distance from the central
            state to ``states[i]``.
        """
        states, distances = self.graph.random_walks(
            width=self.config.n_walks,
            length=self.config.rw_length,
            mode=self.config.rw_mode,
            nbt_history_depth=self.config.nbt_history_depth,
        )
        return states, distances.to(torch.float32)

    def train_step(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> float:
        """Makes one optimizer step on the given batch, then updates the EMA copy of the weights.

        :param states: States (in decoded representation) to train on.
        :param targets: Target distances for `states`.
        :param mask: Which targets are known (optional), see :class:`cayleypy.train.Loss`.
        :param weights: Importance of every target (optional), see :class:`cayleypy.train.Loss`.
        :return: Value of the loss on this batch, before the step.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(self.model(states), targets, mask=mask, weights=weights)
        loss.backward()
        self.optimizer.step()
        self.n_steps += 1
        self._update_ema()
        return float(loss.detach())

    def train_on_data(self, states: torch.Tensor, targets: torch.Tensor) -> float:
        """Makes one pass over the given states, in batches of `TrainConfig.batch_size`.

        :param states: States (in decoded representation) to train on.
        :param targets: Target distances for `states`.
        :return: Mean loss over the batches of this pass.
        """
        n_states = int(states.shape[0])
        if n_states == 0:
            raise ValueError("Cannot train on an empty set of states.")
        n_batches = int(math.ceil(n_states / self.config.batch_size))
        permutation = torch.randperm(n_states, device=states.device)
        losses = [self.train_step(states[idx], targets[idx]) for idx in permutation.tensor_split(n_batches)]
        return sum(losses) / len(losses)

    def train_epoch(self) -> float:
        """Generates data for one epoch and makes one pass over it.

        :return: Mean loss on this epoch.
        """
        learning_rate = self.learning_rate
        states, targets = self.generate_data()
        mean_loss = self.train_on_data(states, targets)
        self.scheduler.step()
        self.epoch += 1
        if self.config.verbose >= 1:
            print(f"Epoch {self.epoch}: loss={mean_loss:.5f}, lr={learning_rate:.3e}.")
        return mean_loss

    def train(self) -> TrainResult:
        """Trains the model for `TrainConfig.n_epochs` epochs.

        Calling this again continues training the same model, but the cosine schedule of the learning rate is built for
        one call, so on the second call the learning rate goes back up.

        :return: Result of training, with the loss curve.
        """
        losses = [self.train_epoch() for _ in range(self.config.n_epochs)]
        if self.config.verbose >= 1:
            print(f"Training finished in {self.n_steps} steps, loss on the last epoch: {losses[-1]:.5f}.")
        return TrainResult(losses=losses, n_steps=self.n_steps)

    def predictor(self, use_ema: bool = True) -> Predictor:
        """Creates predictor using the trained model, e.g. to pass it to beam search.

        :param use_ema: Whether to use the EMA copy of the weights (if it is maintained) rather than the weights
            themselves. Defaults to True.
        :return: The predictor.
        """
        return Predictor(self.graph, self.model_for_inference(use_ema))

    def save(self, path: PathType, use_ema: bool = True) -> ModelConfig:
        """Saves the trained model as a self-describing checkpoint.

        The checkpoint also remembers which graph the model was trained for, so loading it for another graph fails
        instead of silently producing nonsense.

        :param path: Path to the file to write.
        :param use_ema: Whether to save the EMA copy of the weights (if it is maintained) rather than the weights
            themselves. Defaults to True.
        :return: Config stored in the checkpoint (it is `model_config` with the hash of the graph filled in).
        """
        return save_checkpoint(path, self.model_for_inference(use_ema), self.model_config, self.graph.definition)

    def model_for_inference(self, use_ema: bool = True) -> nn.Module:
        """Returns the model to use for prediction - either the EMA copy of the weights, or the weights themselves.

        :param use_ema: Whether to prefer the EMA copy. If it is not maintained (`TrainConfig.ema_decay` is 0), the
            trained model is returned regardless.
        :return: The model.
        """
        if use_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    def _update_ema(self) -> None:
        """Moves the EMA copy of the weights towards the current weights."""
        if self.ema_model is None:
            return
        decay = self.config.ema_decay
        ema_state = self.ema_model.state_dict()
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                ema_value = ema_state[key]
                if ema_value.is_floating_point():
                    ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
                else:
                    # Integer entries (e.g. counters of batch normalization) have no meaningful average.
                    ema_value.copy_(value)
