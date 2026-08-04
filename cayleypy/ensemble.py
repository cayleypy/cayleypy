"""Ensembles of predictors."""

from typing import Optional

import torch

from .models.checkpoint import graph_hash
from .predictor import Predictor


class EnsemblePredictor(Predictor):
    """Predictor computing weighted sum of scores of several other predictors.

    Predictors trained independently make different errors, so their combination estimates distances better than any
    of them alone. In beam search, ensembling two predictors gives improvement comparable to doubling the beam width,
    while being cheaper in memory.

    This ensemble is flat: its members are ordinary predictors, and their scores are combined by a single weighted
    sum. Weights are used as they are given and are not normalized, so it is up to the caller to make them sum to 1
    (which is what the default weights do).

    Both scoring methods are ensembled: :meth:`__call__` combines scores of the states themselves, and
    :meth:`score_children` combines scores of their children (asking every member for its own children scores, so
    members implementing the fast single-pass :meth:`Predictor.score_children` keep using it).

    Members scoring children of a state instead of the state itself (Q-models) can be ensembled through
    :meth:`score_children` only - like the models themselves, they have nothing to say about a state, so
    :meth:`__call__` needs every member to return one score per state.

    Example:

    >>> from cayleypy import CayleyGraph, EnsemblePredictor, PermutationGroups, Predictor
    >>> graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    >>> ensemble = EnsemblePredictor([Predictor(graph, "hamming"), Predictor(graph, "zero")], [0.75, 0.25])
    >>> ensemble(torch.tensor([[1, 0, 2, 3, 4]]))
    tensor([1.5000])
    """

    def __init__(self, members: list[Predictor], weights: Optional[list[float]] = None):
        """Initializes EnsemblePredictor.

        :param members: Predictors to combine. They must be for the same graph, which is checked by comparing the
            mathematical definitions of their graphs (see :func:`cayleypy.models.graph_hash`) - members for different
            graphs would have their scores of unrelated children summed up.
        :param weights: Weight of each member (optional). Score of member ``i`` is multiplied by ``weights[i]``, and
            the products are summed up, without any normalization. If None, defaults to ``1/len(members)`` for every
            member, which makes the ensemble compute the average of its members.
        """
        if len(members) == 0:
            raise ValueError("Ensemble must have at least one member.")
        for i, member in enumerate(members):
            if not isinstance(member, Predictor):
                raise TypeError(
                    f"Member {i} of the ensemble has type {type(member).__name__}, but Predictor was expected. "
                    "Models and heuristics must be wrapped in Predictor first."
                )
        if weights is None:
            weights = [1.0 / len(members)] * len(members)
        elif len(weights) != len(members):
            raise ValueError(f"Ensemble has {len(members)} members, but {len(weights)} weights were given.")

        first = members[0].graph.definition
        first_hash = graph_hash(first)
        for i, member in enumerate(members):
            other = member.graph.definition
            if other.n_generators != first.n_generators or other.state_size != first.state_size:
                raise ValueError(
                    "All members of an ensemble must be for the same graph, but member 0 has "
                    f"{first.n_generators} generators and state size {first.state_size}, while member {i} has "
                    f"{other.n_generators} generators and state size {other.state_size}."
                )
            # Graphs of the same shape can still be different graphs, and for members scoring children even a different
            # order of the generators is enough to make the weighted sum meaningless.
            if graph_hash(other) != first_hash:
                raise ValueError(
                    f"All members of an ensemble must be for the same graph, but member {i} is for another graph with "
                    "the same number of generators and the same state size (its generators or its central state are "
                    "different)."
                )

        self.members = list(members)
        self.weights = [float(w) for w in weights]
        super().__init__(members[0].graph, self._predict_ensemble)

    def _predict_ensemble(self, states: torch.Tensor) -> torch.Tensor:
        ans = self.weights[0] * self.members[0](states)
        for weight, member in zip(self.weights[1:], self.members[1:]):
            ans = ans + weight * member(states)
        return ans

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        """Estimates distances to children of given states as weighted sum of estimates of the members.

        :param states: States (in decoded representation) whose children to score.
        :return: Tensor of shape ``[n_states, n_generators]`` with estimated distances for children.
        """
        ans = self.weights[0] * self.members[0].score_children(states)
        for weight, member in zip(self.weights[1:], self.members[1:]):
            ans = ans + weight * member.score_children(states)
        return ans
