"""Loss functions for training models that estimate distances in a Cayley graph.

Every loss compares predicted distances with target distances and reduces the per-element differences to a single
number. Reduction can ignore unlabeled elements (``mask``) and give elements different importance (``weights``).

Example:

>>> import torch
>>> from cayleypy.train import MseLoss
>>> predictions = torch.tensor([[1.0, 5.0]])
>>> targets = torch.tensor([[2.0, 0.0]])
>>> mask = torch.tensor([[True, False]])
>>> float(MseLoss()(predictions, targets, mask=mask))
1.0
"""

import abc
from typing import Optional

import torch

# Lower bound for the denominator of the weighted mean, so that a batch where nothing is labeled gives 0 instead of
# NaN. Masks are usually indicators, in which case the denominator is a count of labeled elements and never gets
# anywhere near this value.
_MIN_DENOMINATOR = 1e-12


def _check_shape(name: str, tensor: torch.Tensor, predictions: torch.Tensor) -> None:
    if tensor.shape != predictions.shape:
        raise ValueError(
            f"Shape of {name} is {tuple(tensor.shape)}, but it must be the same as shape of predictions, which is "
            f"{tuple(predictions.shape)}."
        )


def _weighted_mean(values: torch.Tensor, mask: Optional[torch.Tensor], weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Computes mean of values, weighted by mask and weights (either of which may be absent)."""
    element_weight: Optional[torch.Tensor] = None
    if mask is not None:
        element_weight = mask.to(values.dtype)
    if weights is not None:
        weights = weights.to(values.dtype)
        element_weight = weights if element_weight is None else element_weight * weights
    if element_weight is None:
        return values.mean()
    return (values * element_weight).sum() / element_weight.sum().clamp_min(_MIN_DENOMINATOR)


class Loss(abc.ABC):
    """Base class for losses used to train distance-estimating models.

    Subclasses define the per-element loss in :meth:`elementwise`. Calling the loss computes those values and reduces
    them to a scalar, which can be backpropagated.

    Losses are not modules (``torch.nn.Module``): they have no learnable parameters, so keeping them out of the module
    tree keeps checkpoints free of entries that mean nothing at inference time.

    Multi-output (Q-) models predict a distance for every generator, i.e. their predictions have shape
    ``[batch_size, n_generators]``, and it is common that only some of these outputs are labeled - a state sampled on
    a random walk has a known target for the generator that produced it, while targets for the other generators are
    unknown. Passing a ``mask`` of the labeled outputs makes both the loss and the gradient see only those, which is
    what makes training on such sparse labels possible.
    """

    @abc.abstractmethod
    def elementwise(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes loss for every element separately, without reduction.

        :param predictions: Predicted distances.
        :param targets: Target distances, of the same shape as ``predictions``.
        :return: Tensor of the same shape as ``predictions``, with loss for every element.
        """

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes mean loss over labeled elements.

        :param predictions: Predicted distances. Shape can be anything, as long as targets, mask and weights have the
            same shape - e.g. ``[batch_size]`` for a model with a single output, or ``[batch_size, n_generators]`` for
            a Q-model.
        :param targets: Target distances, of the same shape as ``predictions``. Values for unlabeled elements (see
            ``mask``) are ignored, so they can be arbitrary.
        :param mask: Which elements are labeled (optional), of the same shape as ``predictions``. Nonzero (or True)
            means "labeled"; unlabeled elements contribute to neither the loss nor the gradient. If None, all elements
            are labeled.
        :param weights: Importance of every element (optional), of the same shape as ``predictions``. Weights are used
            as they are given and are not normalized (the result is the weighted mean, so scaling all of them changes
            nothing). Useful when some targets are less trustworthy than others - for example, lengths of paths found
            by beam search are only upper bounds on the true distance. If None, all elements are equally important.
        :return: Scalar loss - mean of per-element losses over labeled elements, or 0 if nothing is labeled.
        """
        _check_shape("targets", targets, predictions)
        if mask is not None:
            _check_shape("mask", mask, predictions)
        if weights is not None:
            _check_shape("weights", weights, predictions)
        return _weighted_mean(self.elementwise(predictions, targets), mask, weights)


class MseLoss(Loss):
    """Mean squared error loss.

    This is the default way to regress distances. Because the penalty grows quadratically, a few badly predicted
    states matter more than many slightly wrong ones.

    Example:

    >>> import torch
    >>> from cayleypy.train import MseLoss
    >>> float(MseLoss()(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0])))
    0.5
    """

    def elementwise(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes squared error for every element.

        :param predictions: Predicted distances.
        :param targets: Target distances, of the same shape as ``predictions``.
        :return: Tensor of the same shape as ``predictions``, with squared errors.
        """
        return (predictions - targets.to(predictions.dtype)) ** 2


class PinballLoss(Loss):
    """Pinball (quantile) loss, which penalizes underestimation and overestimation differently.

    A model trained with this loss predicts the ``tau``-quantile of the target distribution instead of its mean. With
    ``tau > 0.5``, underestimating a distance costs more than overestimating it by the same amount; with
    ``tau < 0.5``, the other way round, which pushes predictions towards a lower bound on the true distance. At
    ``tau = 0.5`` this is exactly half of the mean absolute error.

    Example:

    >>> import torch
    >>> from cayleypy.train import PinballLoss
    >>> loss = PinballLoss(0.9)
    >>> float(loss(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 2.0])))
    0.5
    """

    def __init__(self, tau: float = 0.5):
        """Initializes PinballLoss.

        :param tau: Quantile of the target distribution to predict. Must be strictly between 0 and 1.
        """
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be strictly between 0 and 1, got {tau}.")
        self.tau = float(tau)

    def elementwise(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes pinball loss for every element.

        :param predictions: Predicted distances.
        :param targets: Target distances, of the same shape as ``predictions``.
        :return: Tensor of the same shape as ``predictions``, with pinball losses - ``tau*(target-prediction)`` where
            the target is underestimated, and ``(1-tau)*(prediction-target)`` where it is overestimated.
        """
        difference = targets.to(predictions.dtype) - predictions
        return torch.maximum(self.tau * difference, (self.tau - 1.0) * difference)


def make_loss(name: str, tau: float = 0.5) -> Loss:
    """Creates loss by name.

    :param name: Name of the loss - either "mse" or "pinball".
    :param tau: Quantile for the pinball loss (ignored by other losses).
    :return: The loss.
    """
    if name == "mse":
        return MseLoss()
    elif name == "pinball":
        return PinballLoss(tau)
    else:
        raise ValueError(f'Unknown loss: "{name}". Supported losses are: "mse", "pinball".')
