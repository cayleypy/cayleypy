import typing
from abc import ABC, abstractmethod

import torch

if typing.TYPE_CHECKING:
    from .cayley_graph import CayleyGraph


class Predictor(ABC):
    """Abstract class representing a "black box" that estimates distance for states from central state."""

    @abstractmethod
    def estimate_distance_to_central_state(self, states: torch.Tensor) -> torch.Tensor:
        """For each state in `states` returns

        :param states: int64 tensor of shape (n, state_size) with states for which we want to estimate distance.
        :return: 1D tensor of length `n` with estimated distances (of any numeric type).
        """

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return self.estimate_distance_to_central_state(states)

    @staticmethod
    def const() -> "ConstPredictor":
        """Always returns 0."""
        return ConstPredictor()

    @staticmethod
    def hamming(graph: "CayleyGraph") -> "HammingPredictor":
        """Heuristic predictor. Estimate is the number of different elements between estimated and central states."""
        return HammingPredictor(graph)


class ConstPredictor(Predictor):
    """Always returns 0."""

    def estimate_distance_to_central_state(self, states: torch.Tensor) -> torch.Tensor:
        return torch.zeros((states.shape[0]))


class HammingPredictor(Predictor):
    """Heuristic predictor. Estimate is the number of different elements between estimated and central states."""

    def __init__(self, graph: "CayleyGraph"):
        self.central_sate = graph.central_state

    def estimate_distance_to_central_state(self, states: torch.Tensor) -> torch.Tensor:
        return torch.not_equal(self.central_sate, states).sum(dim=1)
