import math
import typing
from typing import Callable

import torch

from .models.models_lib import PREDICTOR_MODELS

if typing.TYPE_CHECKING:
    from .cayley_graph import CayleyGraph


def _hamming_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum((x != y), dim=1)


class Predictor:
    """Estimates distance from central state to given states."""

    def __init__(self, graph: "CayleyGraph", models_or_heuristics):
        """Initializes Predictor.

        :param graph: Associated CayleyGraph object.
        :param models_or_heuristics: One of the following:

            - "zero" - will use predictor that returns 0 for any state.
            - "hamming" - will use Hamming distance from central state.
            - ``torch.nn.Module`` - will use given neural network model.
            - Any object that has "predict" method (e.g. sklearn models).
            - Any callable object.
        """
        self.graph = graph
        self.predict = lambda x: x  # type: Callable[[torch.Tensor], torch.Tensor]

        if models_or_heuristics == "zero":
            self.predict = lambda x: torch.zeros((x.shape[0],))
        elif models_or_heuristics == "hamming":
            self.predict = lambda x: _hamming_distance(graph.central_state, x)
        elif isinstance(models_or_heuristics, torch.nn.Module):
            self.predict = models_or_heuristics
            self.predict.eval()
            self.predict.to(graph.device)
        elif hasattr(models_or_heuristics, "predict"):
            self.predict = models_or_heuristics.predict
        elif hasattr(models_or_heuristics, "__call__"):
            self.predict = models_or_heuristics
        else:
            raise ValueError(f"Unable to understand how to call {models_or_heuristics}")

    @staticmethod
    def pretrained(graph: "CayleyGraph"):
        """Loads pre-trained predictor for this graph."""
        if graph.definition.name not in PREDICTOR_MODELS:
            raise KeyError("No pretrained model for this graph.")
        model = PREDICTOR_MODELS[graph.definition.name].load(graph.device)
        return Predictor(graph, model)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        num_batches = int(math.ceil(states.shape[0] / self.graph.batch_size))
        if num_batches > 1:
            ans = []  # type: list[torch.Tensor]
            for batch in states.tensor_split(num_batches, dim=0):
                ans.append(self.predict(batch))
            return torch.hstack(ans)
        else:
            return self.predict(states)
