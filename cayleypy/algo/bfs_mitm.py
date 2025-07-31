"""Breadth-first-search with meet-in-the-middle."""

from typing import Union, Optional

import numpy as np
import torch

from ..bfs_result import BfsResult
from ..cayley_graph import CayleyGraph
from ..torch_utils import isin_via_searchsorted


def find_path_bfs_mitm(
    graph: CayleyGraph,
    start_state: Union[torch.Tensor, np.ndarray, list],
    bfs_result: BfsResult,
) -> Optional[list[int]]:
    """Finds path from ``start_state`` to central state using Meet-in-the-Middle algorithm and precomputed BFS result.

    This algorithm will start BFS from ``start_state`` and for each layer check whether it intersects with already
    found states in ``bfs_result``.

    If shortest path has length ``<= 2*bfs_result.diameter()``, this algorithm is guaranteed to find the shortest path.
    Otherwise, it returns None.

    Works only for inverse-closed generators.

    :param graph: Graph in which path needs to be found.
    :param start_state: First state of the path.
    :param bfs_result: precomputed partial BFS result.
    :return: The found path (list of generator ids), or ``None`` if path was not found.
    """
    assert bfs_result.graph == graph.definition
    assert graph.definition.generators_inverse_closed
    bfs_result.check_has_layer_hashes()
    assert bfs_result.layers_hashes[0][0] == graph.central_state_hash, "Must use the same hasher for bfs_result."

    # First, check if this state is already in bfs_result.
    path = graph.find_path_from(start_state, bfs_result)
    if path is not None:
        return path

    bfs_result.check_has_layer_hashes()
    bfs_last_layer = bfs_result.layers_hashes[-1]
    middle_states = []

    def _stop_condition(layer2, layer2_hashes):
        mask = isin_via_searchsorted(layer2_hashes, bfs_last_layer)
        if not torch.any(mask):
            return False
        for state in graph.decode_states(layer2[mask.nonzero().reshape((-1,))]):
            middle_states.append(state)
        return True

    bfs_result_2 = graph.bfs(
        start_states=start_state,
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
        path1 = graph.restore_path(bfs_result_2.layers_hashes[:-1], middle_state)
        return path1 + graph.definition.revert_path(path2)
    return None
