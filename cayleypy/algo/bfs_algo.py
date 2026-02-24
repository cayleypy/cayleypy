import math
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np
import torch

from cayleypy.algo.bfs_result import BfsResult
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


class BfsAlgorithm:
    """Basic version of the bread-first search (BFS) algorithm."""

    @staticmethod
    def bfs(
        graph: "CayleyGraph",
        *,
        start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**12,
        max_diameter: int = 1000000,
        return_all_edges: bool = False,
        return_all_hashes: bool = False,
        stop_condition: Optional[Callable[[torch.Tensor, torch.Tensor], bool]] = None,
        disable_batching: bool = False,
    ) -> BfsResult:
        """Runs bread-first search (BFS) algorithm from given `start_states`.

        BFS visits all vertices of the graph in layers, where next layer contains vertices adjacent to previous layer
        that were not visited before. As a result, we get all vertices grouped by their distance from the set of initial
        states.

        Depending on parameters below, it can be used to:
          * Get growth function (number of vertices at each BFS layer).
          * Get vertices at some first and last layers.
          * Get all vertices.
          * Get all vertices and edges (i.e. get the whole graph explicitly).

        :param graph: CayleyGraph object on which to run BFS.
        :param start_states: states on 0-th layer of BFS. Defaults to destination state of the graph.
        :param max_layer_size_to_store: maximal size of layer to store.
               If None, all layers will be stored (use this if you need full graph).
               Defaults to 1000.
               First and last layers are always stored.
        :param max_layer_size_to_explore: if reaches layer of larger size, will stop the BFS.
        :param max_diameter: maximal number of BFS iterations.
        :param return_all_edges: whether to return list of all edges (uses more memory).
        :param return_all_hashes: whether to return hashes for all vertices (uses more memory).
        :param stop_condition: function to be called after each iteration. It takes 2 tensors: latest computed layer and
            its hashes, and returns whether BFS must immediately terminate. If it returns True, the layer that was
            passed to the function will be the last returned layer in the result. This function can also be used as a
            "hook" to do some computations after BFS iteration (in which case it must always return False).
        :param disable_batching: Disable batching. Use if you need states and hashes to be in the same order.
        :return: BfsResult object with requested BFS results.
        """
        if start_states is None:
            start_states = graph.central_state
        start_states = graph.encode_states(start_states)
        layer1, layer1_hashes = graph.get_unique_states(start_states)
        layer_sizes = [len(layer1)]
        layers = {0: graph.decode_states(layer1)}
        full_graph_explored = False
        edges_list_starts = []
        edges_list_ends = []
        all_layers_hashes = []
        max_layer_size_to_store = max_layer_size_to_store or 10**15

        # When we don't need edges, we can apply more memory-efficient algorithm with batching.
        # This algorithm finds neighbors in batches and removes duplicates from batches before stacking them.
        do_batching = not return_all_edges and not disable_batching

        # Stores hashes of previous layers, so BFS does not visit already visited states again.
        # If generators are inverse closed, only 2 last layers are stored here.
        seen_states_hashes = [layer1_hashes]

        # Returns mask where 0s are at positions in `current_layer_hashes` that were seen previously.
        def _remove_seen_states(current_layer_hashes: torch.Tensor) -> torch.Tensor:
            ans = ~isin_via_searchsorted(current_layer_hashes, seen_states_hashes[-1])
            for h in seen_states_hashes[:-1]:
                ans &= ~isin_via_searchsorted(current_layer_hashes, h)
            return ans

        # Applies the same mask to states and hashes.
        # If states and hashes are the same thing, it will not create a copy.
        def _apply_mask(states, hashes, mask):
            new_states = states[mask]
            new_hashes = graph.hasher.make_hashes(new_states) if graph.hasher.is_identity else hashes[mask]
            return new_states, new_hashes

        # BFS iteration: layer2 := neighbors(layer1)-layer0-layer1.
        for i in range(1, max_diameter + 1):
            if do_batching and len(layer1) > graph.batch_size:
                num_batches = int(math.ceil(layer1_hashes.shape[0] / graph.batch_size))
                layer2_batches = []  # type: list[torch.Tensor]
                layer2_hashes_batches = []  # type: list[torch.Tensor]
                for layer1_batch in layer1.tensor_split(num_batches, dim=0):
                    layer2_batch = graph.get_neighbors(layer1_batch)
                    layer2_batch, layer2_hashes_batch = graph.get_unique_states(layer2_batch)
                    mask = _remove_seen_states(layer2_hashes_batch)
                    for other_batch_hashes in layer2_hashes_batches:
                        mask &= ~isin_via_searchsorted(layer2_hashes_batch, other_batch_hashes)
                    layer2_batch, layer2_hashes_batch = _apply_mask(layer2_batch, layer2_hashes_batch, mask)
                    layer2_batches.append(layer2_batch)
                    layer2_hashes_batches.append(layer2_hashes_batch)
                layer2_hashes = torch.hstack(layer2_hashes_batches)
                layer2_hashes, _ = torch.sort(layer2_hashes)
                layer2 = layer2_hashes.reshape((-1, 1)) if graph.hasher.is_identity else torch.vstack(layer2_batches)
            else:
                layer1_neighbors = graph.get_neighbors(layer1)
                layer1_neighbors_hashes = graph.hasher.make_hashes(layer1_neighbors)
                if return_all_edges:
                    edges_list_starts += [layer1_hashes.repeat(graph.definition.n_generators)]
                    edges_list_ends.append(layer1_neighbors_hashes)

                layer2, layer2_hashes = graph.get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)
                mask = _remove_seen_states(layer2_hashes)
                layer2, layer2_hashes = _apply_mask(layer2, layer2_hashes, mask)

            if layer2.shape[0] * layer2.shape[1] * 8 > 0.1 * graph.memory_limit_bytes:
                graph.free_memory()
            if return_all_hashes:
                all_layers_hashes.append(layer1_hashes)
            if len(layer2) == 0:
                full_graph_explored = True
                break
            if graph.verbose >= 2:
                print(f"Layer {i}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            if len(layer2) <= max_layer_size_to_store:
                layers[i] = graph.decode_states(layer2)

            layer1 = layer2
            layer1_hashes = layer2_hashes
            seen_states_hashes.append(layer2_hashes)
            if graph.definition.generators_inverse_closed:
                # Only keep hashes for last 2 layers.
                seen_states_hashes = seen_states_hashes[-2:]
            if len(layer2) >= max_layer_size_to_explore:
                break
            if stop_condition is not None and stop_condition(layer2, layer2_hashes):
                break

        if return_all_hashes and not full_graph_explored:
            all_layers_hashes.append(layer1_hashes)

        if not full_graph_explored and graph.verbose > 0:
            print("BFS stopped before graph was fully explored.")

        edges_list_hashes: Optional[torch.Tensor] = None
        if return_all_edges:
            if not full_graph_explored:
                # Add copy of edges between last 2 layers, but in opposite direction.
                # This is done so adjacency matrix is symmetric.
                v1, v2 = edges_list_starts[-1], edges_list_ends[-1]
                edges_list_starts.append(v2)
                edges_list_ends.append(v1)
            edges_list_hashes = torch.vstack([torch.hstack(edges_list_starts), torch.hstack(edges_list_ends)]).T

        # Always store the last layer.
        last_layer_id = len(layer_sizes) - 1
        if full_graph_explored and last_layer_id not in layers:
            layers[last_layer_id] = graph.decode_states(layer1)

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=layers,
            bfs_completed=full_graph_explored,
            layers_hashes=all_layers_hashes,
            edges_list_hashes=edges_list_hashes,
            graph=graph.definition,
        )
