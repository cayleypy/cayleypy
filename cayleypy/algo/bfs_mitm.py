"""Breadth-first-search with meet-in-the-middle."""

from typing import Union, Optional

import numpy as np
import torch

from ..bfs_result import BfsResult
from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import AnyStateType
from ..torch_utils import isin_via_searchsorted


class MeetInTheMiddle:
    """Meet-in-the middle (MITM) algorithm for path finding."""

    @staticmethod
    def find_path_to(
        graph: CayleyGraph,
        dest_state: AnyStateType,
        bfs_result: BfsResult,
    ) -> Optional[list[int]]:
        """Finds path from central state to ``dest_state`` using MITM algorithm and precomputed BFS result.

        This algorithm will start BFS from ``dest_state`` and for each layer check whether it intersects with already
        found states in ``bfs_result``. This BFS is done in inverted graph (graph where generators are inverses of
        generators in the original graph).

        If shortest path has length ``<= 2*bfs_result.diameter()``, this algorithm is guaranteed to find the shortest
        path. Otherwise, it returns None.

        :param graph: Graph in which path needs to be found.
        :param dest_state: Last state of the path.
        :param bfs_result: precomputed partial BFS result.
        :return: The found path (list of generator ids), or ``None`` if path was not found.
        """
        assert bfs_result.graph == graph.definition
        bfs_result.check_has_layer_hashes()
        assert bfs_result.layers_hashes[0][0] == graph.central_state_hash, "Must use the same hasher for bfs_result."

        # First, check if this state is already in bfs_result.
        path = graph.find_path_to(dest_state, bfs_result)
        if path is not None:
            return path

        bfs_last_layer = bfs_result.layers_hashes[-1]
        middle_states = []

        def _stop_condition(layer2, layer2_hashes):
            mask = isin_via_searchsorted(layer2_hashes, bfs_last_layer)
            if not torch.any(mask):
                return False
            for state in graph.decode_states(layer2[mask.nonzero().reshape((-1,))]):
                middle_states.append(state)
            return True

        graph_inv = graph.with_inverted_generators
        bfs_result_2 = graph_inv.bfs(
            start_states=dest_state,
            max_diameter=bfs_result.diameter(),
            return_all_hashes=True,
            stop_condition=_stop_condition,
            disable_batching=True,
        )

        if len(middle_states) == 0:
            return None

        for middle_state in middle_states:
            try:
                path2 = graph.restore_path(bfs_result.layers_hashes[:-1], middle_state)
            except AssertionError as ex:
                print("Warning! State did not work due to hash collision!", ex)
                continue
            path1 = graph_inv.restore_path(bfs_result_2.layers_hashes[:-1], middle_state)
            return path2 + path1[::-1]
        return None

    @staticmethod
    def find_path_from(
        graph: CayleyGraph,
        start_state: Union[torch.Tensor, np.ndarray, list],
        bfs_result: BfsResult,
    ) -> Optional[list[int]]:
        """Finds path from ``start_state`` to central state using MITM algorithm and precomputed BFS result.

        This is a wrapper around ``find_path_to`` and it works only for inverse-closed generators.
        """
        assert graph.definition.generators_inverse_closed
        path = MeetInTheMiddle.find_path_to(graph, start_state, bfs_result)
        if path is None:
            return None
        return graph.definition.revert_path(path)

    @staticmethod
    def find_path_between(
        graph: CayleyGraph,
        start_state: AnyStateType,
        dest_state: AnyStateType,
        max_diameter: int = 10,
    ):
        """Finds shortest path between two states using MITM algorithm.

        If shortest path has length ``<= 2*max_diameter``, this algorithm is guaranteed to find the shortest
        path. Otherwise, it returns None.

        :param graph: Graph in which path needs to be found.
        :param start_state: First state of the path.
        :param dest_state: Last state of the path.
        :param max_diameter: depth of BFS.
        :return: The found path (list of generator ids), or ``None`` if path was not found.
        """
        graph1 = graph.modified_copy(graph.definition.with_central_state(start_state))
        bfs_result = graph1.bfs(max_diameter=max_diameter, return_all_hashes=True, max_layer_size_to_store=0)
        assert bfs_result.layers_hashes[0][0] == graph1.central_state_hash
        return MeetInTheMiddle.find_path_to(graph1, dest_state, bfs_result)
