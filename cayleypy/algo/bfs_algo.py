import math
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np
import torch

from ..bfs_result import BfsResult
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
        """Runs bread-first search (BFS) algorithm from given `start_states`."""
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

        do_batching = not return_all_edges and not disable_batching
        seen_states_hashes = [layer1_hashes]

        def _remove_seen_states(current_layer_hashes: torch.Tensor) -> torch.Tensor:
            ans = ~isin_via_searchsorted(current_layer_hashes, seen_states_hashes[-1])
            for h in seen_states_hashes[:-1]:
                ans &= ~isin_via_searchsorted(current_layer_hashes, h)
            return ans

        def _apply_mask(states, hashes, mask):
            new_states = states[mask]
            new_hashes = graph.hasher.make_hashes(new_states) if graph.hasher.is_identity else hashes[mask]
            return new_states, new_hashes

        for i in range(1, max_diameter + 1):
            if do_batching and len(layer1) > graph.batch_size:
                num_batches = int(math.ceil(layer1_hashes.shape[0] / graph.batch_size))
                layer2_batches = []
                layer2_hashes_batches = []
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
                v1, v2 = edges_list_starts[-1], edges_list_ends[-1]
                edges_list_starts.append(v2)
                edges_list_ends.append(v1)
            edges_list_hashes = torch.vstack([torch.hstack(edges_list_starts), torch.hstack(edges_list_ends)]).T

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
