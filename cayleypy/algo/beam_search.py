"""Beam search algorithm for Cayley graphs."""

from typing import TYPE_CHECKING, Optional

import torch

from ..beam_search_result import BeamSearchResult
from ..bfs_result import BfsResult
from ..cayley_graph_def import AnyStateType
from ..predictor import Predictor
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


class BeamSearchAlgorithm:
    """Beam search algorithm for finding paths in Cayley graphs.

    This class implements the beam search algorithm to find paths from a given start state
    to the central state of a Cayley graph. It can use various heuristics (predictors) to
    guide the search and supports meet-in-the-middle optimization.
    """

    def __init__(self, graph: "CayleyGraph"):
        """Initialize the beam search algorithm.

        :param graph: The Cayley graph to search on.
        """
        self.graph = graph

    def search(
        self,
        *,
        start_state: AnyStateType,
        predictor: Optional[Predictor] = None,
        beam_width=1000,
        max_iterations=1000,
        return_path=False,
        bfs_result_for_mitm: Optional[BfsResult] = None,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to central state using Beam Search algorithm.

        :param start_state: State from which to start search.
        :param predictor: A heuristic that estimates scores for states (lower score = closer to center).
          Defaults to Hamming distance heuristic.
        :param beam_width: Width of the beam (how many "best" states we consider at each step).
        :param max_iterations: Maximum number of iterations before giving up.
        :param return_path: Whether to return path (consumes much more memory if True).
        :param bfs_result_for_mitm: BfsResult with pre-computed neighborhood of central state to compute for
            meet-in-the-middle modification of Beam Search. Beam search will terminate when any of states in that
            neighborhood is encountered. Defaults to None, which means no meet-in-the-middle (i.e. only search for the
            central state).
        :return: BeamSearchResult containing found path length and (optionally) the path itgraph.
        """
        graph = self.graph
        if predictor is None:
            predictor = Predictor(graph, "hamming")

        start_states = graph.encode_states(start_state)
        layer1, layer1_hashes = graph.get_unique_states(start_states)
        all_layers_hashes = [layer1_hashes]
        debug_scores = {}  # type: dict[int, float]

        if graph.central_state_hash[0] == layer1_hashes[0]:
            # Start state is the central state.
            return BeamSearchResult(True, 0, [], debug_scores, graph.definition)

        bfs_layers_hashes = [graph.central_state_hash]
        if bfs_result_for_mitm is not None:
            assert bfs_result_for_mitm.graph == graph.definition
            bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Checks if any of `hashes` are in neighborhood of the central state.
        # Returns the number of the first layer where intersection was found, or -1 if not found.
        def _check_path_found(hashes):
            for j, layer in enumerate(bfs_layers_hashes):
                if torch.any(isin_via_searchsorted(layer, hashes)):
                    return j
            return -1

        def _restore_path(found_layer_id: int) -> Optional[list[int]]:
            if not return_path:
                return None
            if found_layer_id == 0:
                return graph.restore_path(all_layers_hashes, graph.central_state)
            assert bfs_result_for_mitm is not None
            mask = isin_via_searchsorted(layer2_hashes, bfs_layers_hashes[found_layer_id])
            assert torch.any(mask), "No intersection in Meet-in-the-middle"
            middle_state = graph.decode_states(layer2[mask.nonzero()[0].item()].reshape((1, -1)))
            path1 = graph.restore_path(all_layers_hashes, middle_state)
            path2 = graph.find_path_from(middle_state, bfs_result_for_mitm)
            assert path2 is not None
            return path1 + path2

        for i in range(max_iterations):
            # Create states on the next layer.
            layer2, layer2_hashes = graph.get_unique_states(graph.get_neighbors(layer1))

            bfs_layer_id = _check_path_found(layer2_hashes)
            if bfs_layer_id != -1:
                # Path found.
                path = _restore_path(bfs_layer_id)
                return BeamSearchResult(True, i + bfs_layer_id + 1, path, debug_scores, graph.definition)

            # Pick `beam_width` states with lowest scores.
            if len(layer2) >= beam_width:
                scores = predictor(graph.decode_states(layer2))
                idx = torch.argsort(scores)[:beam_width]
                layer2 = layer2[idx, :]
                layer2_hashes = layer2_hashes[idx]
                best_score = float(scores[idx[0]])
                debug_scores[i] = best_score
                if graph.verbose >= 2:
                    print(f"Iteration {i}, best score {best_score}.")

            layer1 = layer2
            layer1_hashes = layer2_hashes
            if return_path:
                all_layers_hashes.append(layer1_hashes)

        # Path not found.
        return BeamSearchResult(False, 0, None, debug_scores, graph.definition)
