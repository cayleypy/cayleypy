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
        fastest, meaning the number of steps is the closest estimate of the true distance.
    :param nbt_history_depth: For "nbt" mode, how many previous levels to remember and ban from revisiting. Must be at
        least 1 in that mode, because a walk that bans nothing never counts a step.
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
        if self.rw_mode == "nbt" and self.nbt_history_depth == 0:
            # The step counter of a non-backtracking walk only advances when the walk moves to a state that is not
            # banned, so with nothing banned it never advances and every generated state gets target distance 0.
            raise ValueError(
                'nbt_history_depth must be at least 1 in "nbt" mode, got 0. A walk that remembers no previous levels '
                "never counts a step, so every state it generates would be labelled with distance 0."
            )
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
