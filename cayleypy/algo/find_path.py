"""Automatic path finding."""

import time
from typing import Optional

from .bfs_mitm import find_path_bfs_mitm
from ..predictor import Predictor
from ..bfs_result import BfsResult
from ..cayley_graph import CayleyGraph, AnyStateType
from ..models.models_lib import PREDICTOR_MODELS


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
    """kashfkjah

    If you want to compute multiple results, pass the same `graph` object so some computatoions can be rused.

    Use specialized.
    """

    # If we have pre-trained model for beam search, use beam search with that predictor.
    if graph.definition.name in PREDICTOR_MODELS:
        predictor = Predictor.pretrained(graph)
        result = graph.beam_search(
            start_state=start_state,
            predictor=predictor,
            beam_width=kwargs.get("beam_width") or 10**4,
            max_iterations=kwargs.get("max_iterations") or 10**9,
            return_path=True,
        )
        return result.path

    # Try finding exact solution using pre-computed cached BFS result.
    # This will work for small graphs or short paths.
    # If this fails, we return None (for now). In future more path finding algorithms might be added.
    bfs_result = _precompute_bfs(graph, **kwargs)
    if graph.definition.generators_inverse_closed:
        return find_path_bfs_mitm(graph, start_state, bfs_result)
    else:
        return graph.find_path_from(start_state, bfs_result)
