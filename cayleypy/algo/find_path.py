"""Automatic path finding."""

import time
from typing import Optional

from .bfs_mitm import MeetInTheMiddle
from ..bfs_result import BfsResult
from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import AnyStateType
from ..models.models_lib import PREDICTOR_MODELS
from ..predictor import Predictor


def _precompute_bfs(graph: CayleyGraph, **kwargs) -> BfsResult:
    """Computes BfsResult of reasonable size to assist with path finding and caches it."""
    if not hasattr(graph, "_bfs_result_for_find_path"):
        if graph.verbose > 0:
            print(f"Pre-computing bfs for {graph.definition.name}...")
        t0 = time.time()
        result = graph.bfs(
            max_layer_size_to_store=0,
            max_layer_size_to_explore=kwargs.get("max_layer_size_to_explore") or 10**6,
            max_diameter=kwargs.get("max_diameter") or 50,
            return_all_hashes=True,
        )
        time_delta = time.time() - t0
        if graph.verbose > 0:
            print(f"Pre-computed BFS with {result.diameter()} layers in {time_delta:.02f}s.")
        setattr(graph, "_bfs_result_for_find_path", result)
    return getattr(graph, "_bfs_result_for_find_path")


def find_path(graph: CayleyGraph, start_state: AnyStateType, **kwargs) -> Optional[list[int]]:
    """Finds path from ``start_state`` to central state.

    This function will try to automatically pick the best algorithm. It will return ``None`` if path was not found,
    which can happen if path does not exist, or if it exists but the algorithm failed to find it.

    The path is returned as list of generator numbers. Use ``graph.definition.path_to_string(path)`` to convert it to
    string.

    If you want to compute paths from multiple different states for the same graph, use the same instance of ``graph``
    and pass it here. This will allow doing some computations only once (on the first call) and caching the results,
    making subsequent path computations faster.

    In practice, this will work only for very small graphs, very short paths, and for graphs for which the library has
    pre-computed ML models. Also, there is no guarantee that the path fill be the shortest (although this function will
    try to find short path). Therefore, it's recommended to use specialized algorithms for path finding instead of this
    function.

    :param graph: Graph in which to find the path.
    :param start_state: First state of the path.
    :return: The found path (list of generator ids), or ``None`` if path was not found.
    """

    # If we have pre-trained model for beam search, use beam search with that predictor.
    if graph.definition.name in PREDICTOR_MODELS:
        predictor = Predictor.pretrained(graph)
        result = graph.beam_search(
            start_state=start_state,
            predictor=predictor,
            beam_width=kwargs.get("beam_width") or 10**4,
            max_steps=kwargs.get("max_steps") or 10**9,
            return_path=True,
            **kwargs,
        )
        return result.path

    # Try finding exact solution using pre-computed cached BFS result.
    # This will work for small graphs or short paths.
    # If this fails, we return None (for now). In future more path finding algorithms might be added.
    if graph.definition.generators_inverse_closed:
        bfs_result = _precompute_bfs(graph, **kwargs)
        return MeetInTheMiddle.find_path_from(graph, start_state, bfs_result)
    else:
        graph_inv = graph.with_inverted_generators
        bfs_result = _precompute_bfs(graph_inv, **kwargs)
        path = MeetInTheMiddle.find_path_to(graph_inv, start_state, bfs_result)
        if path is None:
            return None
        return path[::-1]
