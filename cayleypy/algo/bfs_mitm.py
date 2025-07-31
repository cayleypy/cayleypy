"""Breadth-first-search with meet-in-the-middle."""

from typing import Union, Optional

import numpy as np
import torch

from ..bfs_result import BfsResult
from ..cayley_graph import CayleyGraph
from ..permutation_utils import permutation_between
from ..torch_utils import isin_via_searchsorted


def _bfs_mitm_optimized(
    graph: CayleyGraph,
    start_state: Union[torch.Tensor, np.ndarray, list],
    bfs_result: BfsResult,
) -> Optional[list[int]]:
    """Optimized Meet-in-the-Middle.

    This algorithm doesn't compute BFS from `start_state`. Instead it takes pre-computed bfs_result from start state and
    applies a permutation to it which produces exactly the same result.
    """
    assert graph.definition.is_permutation_group()
    assert graph.string_encoder is not None
    assert graph.hasher.is_identity
    bfs_result.check_has_layer_hashes()
    bfs_last_layer = bfs_result.layers_hashes[-1]

    perm = permutation_between(graph.definition.central_state, start_state)
    perm_func = graph.string_encoder.implement_permutation_1d(perm)
    bfs_result_2_layers_hashes = []

    print("perm_func(central):", perm_func(graph.central_state_hash))
    print("start_state", graph.encode_states(start_state))

    for i, layer in enumerate(bfs_result.layers_hashes):
        layer2 = perm_func(layer)
        layer2, _ = torch.sort(layer2)
        if len(layer2) < 5:
            print("layer2", i, layer2)
        bfs_result_2_layers_hashes.append(layer2)
        mask = isin_via_searchsorted(bfs_last_layer, layer2)
        if torch.any(mask):
            middle_state = graph.decode_states(bfs_last_layer[mask.nonzero()[0].item()].reshape((1, -1)))
            path1 = graph.restore_path(bfs_result_2_layers_hashes[:-1], middle_state)
            path2 = graph.restore_path(bfs_result.layers_hashes[:-1], middle_state)
            return path1 + graph.definition.revert_path(path2)
    return None


def find_path_bfs_mitm(
    graph: CayleyGraph,
    start_state: Union[torch.Tensor, np.ndarray, list],
    bfs_result: BfsResult,
    tmp_use_new_algo=False,
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

    # If this is permutation graph and states fit in single int64, can run optimized version of Meet-in-the-Middle.
    if (
        graph.definition.is_permutation_group()
        and graph.string_encoder is not None
        and graph.hasher.is_identity
        and tmp_use_new_algo
    ):
        return _bfs_mitm_optimized(graph, start_state, bfs_result)

    bfs_last_layer = bfs_result.layers_hashes[-1]
    middle_states = []

    def _stop_condition(layer2, layer2_hashes):
        if len(layer2) < 5:
            print("SC", layer2)
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
