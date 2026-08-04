"""Lower bounds on distance to the central state, used to prune beam search."""

from typing import Protocol, TYPE_CHECKING, runtime_checkable

import torch

from .algo.bfs_result import BfsResult

if TYPE_CHECKING:
    from .cayley_graph import CayleyGraph


@runtime_checkable
class LowerBound(Protocol):
    """Lower bound on distance from a state to the central state.

    The bound must be **admissible**: for every state, the returned value must not exceed the true distance from that
    state to the central state. Beam search uses it to drop states that cannot be on a path shorter than a known upper
    bound (see ``lower_bound`` and ``prune_above`` in
    :meth:`cayleypy.algo.BeamSearchAlgorithm.search_simple`). A bound that is not admissible can make the search miss
    paths that it would otherwise find.

    This is a protocol: any object having the ``lb`` method can be used, it does not have to inherit from this class.
    """

    def lb(self, states: torch.Tensor) -> torch.Tensor:
        """Estimates lower bounds on distances from given states to the central state.

        :param states: One state (1-D tensor) or multiple states (2-D tensor), in decoded representation.
        :return: Tensor of shape ``[n_states]`` with a lower bound for each state.
        """


class BfsLowerBound:
    """Lower bound based on exact distances in a pre-computed ball around the central state.

    BFS from the central state gives exact distances for all states within some radius ``d`` of it. For a state inside
    that ball, the exact distance is returned, which is the strongest possible bound. Every other state is at distance
    at least ``d+1``, and that is what is returned for it. The larger the ball, the stronger the bound - and the more
    memory it takes (one hash per state in the ball).

    Generators must be inverse-closed, so that distance from the central state to a state is equal to distance from
    that state back to the central state.

    Example:

    >>> from cayleypy import BfsLowerBound, CayleyGraph, PermutationGroups
    >>> graph = CayleyGraph(PermutationGroups.lrx(6))
    >>> lower_bound = BfsLowerBound(graph, graph.bfs(max_diameter=2, return_all_hashes=True))
    >>> lower_bound.lb([[0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]).tolist()
    [0, 1, 3]
    """

    def __init__(self, graph: "CayleyGraph", bfs_result: BfsResult):
        """Initializes BfsLowerBound.

        :param graph: The Cayley graph. Distances are to the central state of this graph.
        :param bfs_result: Result of BFS from the central state of `graph`, computed with ``return_all_hashes=True``.
            It must be computed by the same `CayleyGraph` object, because states are looked up by hash, and hashing
            depends on a random seed stored in the graph.
        """
        if not graph.definition.generators_inverse_closed:
            raise ValueError(
                "BfsLowerBound requires inverse-closed generators (for every generator, its inverse must also be a "
                "generator), otherwise distances computed by BFS from the central state are distances in the wrong "
                "direction."
            )
        if bfs_result.graph != graph.definition:
            raise ValueError("BFS was run on a different graph.")
        layers_hashes = bfs_result.layers_hashes
        if len(layers_hashes) != len(bfs_result.layer_sizes):
            raise ValueError("Hashes for some layers are missing. Run bfs with return_all_hashes=True.")
        if len(layers_hashes[0]) != 1 or int(layers_hashes[0][0]) != int(graph.central_state_hash[0]):
            raise ValueError(
                "BFS must be started from the central state (which is the default), and must be run by the same "
                "CayleyGraph object, because states are looked up by hash."
            )
        device = graph.device
        hashes = torch.cat([h.reshape(-1).to(device) for h in layers_hashes])
        distances = torch.cat(
            [torch.full((len(h),), i, dtype=torch.int64, device=device) for i, h in enumerate(layers_hashes)]
        )
        order = torch.argsort(hashes)
        self.graph = graph
        self.radius = len(layers_hashes) - 1
        """Radius of the ball for which exact distances are known."""
        self._hashes = hashes[order]
        self._distances = distances[order]

    def lb(self, states: torch.Tensor) -> torch.Tensor:
        """Returns exact distances for states within the ball, and ``radius+1`` for all other states.

        :param states: One state (1-D tensor) or multiple states (2-D tensor), in decoded representation.
        :return: Tensor of shape ``[n_states]`` with a lower bound for each state.
        """
        hashes = self.graph.hasher.make_hashes(self.graph.encode_states(states))
        idx = torch.searchsorted(self._hashes, hashes).clamp(max=self._hashes.shape[0] - 1)
        found = self._hashes[idx] == hashes
        outside = torch.full_like(hashes, self.radius + 1)
        return torch.where(found, self._distances[idx], outside)
