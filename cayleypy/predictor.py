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

    def predict_batched(self, states: torch.Tensor) -> torch.Tensor:
        """Applies the underlying model to `states`, splitting them into batches if there are too many.

        Output of the model is returned as is. It has shape ``[n_states]`` for usual (single-output) models, and shape
        ``[n_states, n_outputs]`` for multi-output models (e.g. models predicting one score per generator).

        :param states: States (in decoded representation) to apply the model to.
        :return: Output of the model for `states`.
        """
        # A predictor is only ever used for inference, and building the graph for a backward pass that never happens
        # costs memory proportional to the size of the model - which is significant when the whole beam is scored.
        with torch.no_grad():
            num_batches = int(math.ceil(states.shape[0] / self.graph.batch_size))
            if num_batches > 1:
                ans = []  # type: list[torch.Tensor]
                for batch in states.tensor_split(num_batches, dim=0):
                    ans.append(self.predict(batch))
                # Batches must be concatenated along dimension 0, otherwise outputs of multi-output models are mangled.
                return torch.cat(ans, dim=0)
            else:
                return self.predict(states)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        ans = self.predict_batched(states)
        if len(ans.shape) != 1:
            raise ValueError(
                f"Model returned output of shape {tuple(ans.shape)}, but one score per state (1-D output) was "
                "expected. Models having one output per generator must implement Predictor.score_children instead."
            )
        return ans

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        """Estimates distances from central state to all children (neighbors) of given states.

        Children are enumerated in the order of generators: element ``[i, j]`` of the answer is the estimated distance
        for the state obtained by applying generator ``j`` to ``states[i]``.

        This default implementation calls the underlying model for every child, so it needs `n_generators` times more
        model evaluations than :meth:`__call__`. Models having one output per generator (Q-models) are expected to
        override this method and compute the whole answer in a single forward pass.

        :param states: States (in decoded representation) whose children to score.
        :return: Tensor of shape ``[n_states, n_generators]`` with estimated distances for children.
        """
        n_generators = self.graph.definition.n_generators
        encoded_states = self.graph.encode_states(states)
        num_states = int(encoded_states.shape[0])
        children = self.graph.decode_states(self.graph.get_neighbors(encoded_states))
        scores = self(children)
        # `get_neighbors` returns neighbors in generator-major order: rows [i*num_states, (i+1)*num_states) are children
        # obtained by applying generator i. Hence, the answer must be transposed.
        return scores.reshape((n_generators, num_states)).transpose(0, 1).contiguous()
