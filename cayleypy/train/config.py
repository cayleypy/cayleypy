"""Configuration of the training process."""

from dataclasses import dataclass
from typing import Optional

from .losses import Loss, make_loss

# Modes of random walk generation that can be used to produce training data, see
# :class:`cayleypy.algo.RandomWalksGenerator`.
RANDOM_WALK_MODES = ("classic", "bfs", "nbt")


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration of the process of training a model to estimate distances in a Cayley graph.

    Training is split into epochs. On every epoch, a fresh set of random walks is generated and the model makes one
    pass over it in batches of `batch_size` states. Because data is regenerated, the model almost never sees the same
    state twice, so there is no separate notion of a dataset.

    Defaults are chosen to train a small model on a small graph in seconds. Serious training needs longer walks, more
    of them, and more epochs.

    :param n_epochs: Number of epochs (how many times to generate data and pass over it).
    :param n_walks: Number of random walks generated on every epoch (`width` of the walks).
    :param rw_length: Length of every random walk. Must be at least 2. Should be at least as large as the diameter of
        the graph, otherwise the model never sees distant states.
    :param rw_mode: Mode of random walk generation - one of "classic", "bfs", "nbt". Defaults to "nbt", which mixes
        fastest, meaning the number of steps is the closest estimate of the true distance. Ignored when training a
        Q-model: its data comes from :class:`cayleypy.train.SparseQSampler`, which needs walks to be paths and
        therefore always uses "classic" walks.
    :param nbt_history_depth: For "nbt" mode, how many previous levels to remember and ban from revisiting.
    :param anchors_depth: Depth of the breadth-first search producing anchors - states with exact distances that are
        mixed into the data, see :class:`cayleypy.train.BfsAnchors`. 0 (the default) means no anchors. Note that memory
        needed for the search grows quickly with this depth.
    :param anchors_fraction: Share of anchors in the data of one epoch (ignored if `anchors_depth` is 0, but still
        required to be strictly between 0 and 1 - to train without anchors, leave `anchors_depth` at 0 and this field at
        its default; :class:`cayleypy.train.BellmanTrainer` always uses it, because anchors are mandatory there). A few
        per cent is what helps; a large share (10% and more, empirically) makes the model good near the central state and
        worse where beam search actually spends its time.
    :param bellman_anchors_depth: Depth of the breadth-first search producing anchors for
        :class:`cayleypy.train.BellmanTrainer`, which needs them (unlike the walk-based trainer, where `anchors_depth`
        is 0 by default): bootstrapped targets only say how far states are from each other, so without exactly known
        distances the scale of the predictions drifts. Must be at least 1, which means the central state and its
        neighbors. Ignored when training on walk targets.
    :param bellman_target_update_period: How many epochs :class:`cayleypy.train.BellmanTrainer` keeps the target (the
        frozen copy of the model that computes targets) before refreshing it from the model being trained. 1 (the
        default) means the target is the EMA copy of the weights, which lags behind by design; a larger value holds it
        fixed for several epochs instead. Ignored when training on walk targets.
    :param batch_size: Number of states in one training batch.
    :param lr: Initial learning rate.
    :param lr_min: Learning rate at the end of training. The learning rate follows a cosine schedule from `lr` to
        `lr_min` over `n_epochs` epochs.
    :param weight_decay: Weight decay of the AdamW optimizer. 0 (the default) makes it behave as plain Adam.
    :param ema_decay: Decay of the exponential moving average of weights, which is usually a better predictor than the
        weights themselves. 0 disables averaging.
    :param loss: Name of the loss - see :func:`cayleypy.train.make_loss`.
    :param tau: Quantile for the pinball loss (ignored by other losses).
    :param seed: Random seed. If set, training is deterministic on a given device.
    :param verbose: Level of logging. 0 means no logging, 1 means one line per epoch.
    """

    n_epochs: int = 100
    n_walks: int = 128
    rw_length: int = 20
    rw_mode: str = "nbt"
    nbt_history_depth: int = 1
    anchors_depth: int = 0
    anchors_fraction: float = 0.02
    bellman_anchors_depth: int = 1
    bellman_target_update_period: int = 1
    batch_size: int = 512
    lr: float = 1e-3
    lr_min: float = 0.0
    weight_decay: float = 0.0
    ema_decay: float = 0.99
    loss: str = "mse"
    tau: float = 0.5
    seed: Optional[int] = None
    verbose: int = 0

    def __post_init__(self):
        _check_positive("n_epochs", self.n_epochs)
        _check_positive("n_walks", self.n_walks)
        _check_positive("batch_size", self.batch_size)
        _check_positive("lr", self.lr)
        if self.rw_length < 2:
            raise ValueError(f"rw_length must be at least 2, got {self.rw_length}.")
        if self.rw_mode not in RANDOM_WALK_MODES:
            raise ValueError(f'Unknown rw_mode: "{self.rw_mode}". Supported modes are: {RANDOM_WALK_MODES}.')
        if self.nbt_history_depth < 0:
            raise ValueError(f"nbt_history_depth must be non-negative, got {self.nbt_history_depth}.")
        if self.anchors_depth < 0:
            raise ValueError(f"anchors_depth must be non-negative, got {self.anchors_depth}.")
        if not 0.0 < self.anchors_fraction < 1.0:
            raise ValueError(f"anchors_fraction must be strictly between 0 and 1, got {self.anchors_fraction}.")
        if self.bellman_anchors_depth < 1:
            raise ValueError(
                f"bellman_anchors_depth must be at least 1, got {self.bellman_anchors_depth}. Bellman targets are "
                "bootstrapped from the model itself, so anchors with exactly known distances are what keeps the scale "
                "of the predictions from drifting, and cannot be turned off."
            )
        _check_positive("bellman_target_update_period", self.bellman_target_update_period)
        if not 0.0 <= self.lr_min <= self.lr:
            raise ValueError(f"lr_min must be between 0 and lr={self.lr}, got {self.lr_min}.")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be at least 0 and less than 1, got {self.ema_decay}.")
        # Creates the loss to fail early (rather than after the first epoch of data generation) if it is misconfigured.
        self.make_loss()

    def make_loss(self) -> Loss:
        """Creates the loss described by this config."""
        return make_loss(self.loss, self.tau)
