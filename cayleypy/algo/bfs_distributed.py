import math
from typing import Callable, Optional, TYPE_CHECKING, Union

import numpy as np
import torch

from .bfs_result import BfsResult
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


LayerPart = tuple[torch.Tensor, torch.Tensor]


class BfsDistributed:
    """Multi-GPU breadth-first search implementation."""

    @staticmethod
    def _empty_part(device: torch.device, state_width: int) -> LayerPart:
        return (
            torch.empty((0, state_width), dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )

    @staticmethod
    def _apply_mask(states: torch.Tensor, hashes: torch.Tensor, mask: torch.Tensor) -> LayerPart:
        return states[mask], hashes[mask]

    @classmethod
    def _partition_states(cls, graph: "CayleyGraph", states: torch.Tensor, hashes: torch.Tensor) -> list[LayerPart]:
        parts = []
        for owner, device in enumerate(graph.gpu_devices):
            mask = (hashes % graph.num_gpus) == owner
            part_states = states[mask].to(device)
            part_hashes = hashes[mask].to(device)
            if len(part_hashes) > 0:
                part_hashes, idx = torch.sort(part_hashes, stable=True)
                part_states = part_states[idx]
            parts.append((part_states, part_hashes))
        return parts

    @staticmethod
    def _gather_parts(graph: "CayleyGraph", layer_parts: list[LayerPart]) -> LayerPart:
        all_hashes = torch.cat([hashes.to(graph.device) for _, hashes in layer_parts], dim=0)
        if len(all_hashes) == 0:
            return (
                torch.empty((0, graph.encoded_state_size), dtype=torch.int64, device=graph.device),
                torch.empty(0, dtype=torch.int64, device=graph.device),
            )
        all_states = torch.cat([states.to(graph.device) for states, _ in layer_parts], dim=0)
        all_hashes, idx = torch.sort(all_hashes, stable=True)
        return all_states[idx], all_hashes

    @staticmethod
    def _update_seen_parts(
        graph: "CayleyGraph",
        seen_parts: list[list[torch.Tensor]],
        previous_parts: list[LayerPart],
        next_parts: list[LayerPart],
    ) -> None:
        for owner in range(graph.num_gpus):
            prev_hashes = previous_parts[owner][1]
            next_hashes = next_parts[owner][1]
            if graph.definition.generators_inverse_closed:
                seen_parts[owner] = [part for part in [prev_hashes, next_hashes] if len(part) > 0]
            elif len(next_hashes) > 0:
                seen_parts[owner].append(next_hashes)

    @staticmethod
    def _remove_seen_states(
        graph: "CayleyGraph",
        seen_parts: list[list[torch.Tensor]],
        current_layer_hashes: torch.Tensor,
        streams: list[torch.cuda.Stream],
    ) -> torch.Tensor:
        device_masks = []
        for owner, device in enumerate(graph.gpu_devices):
            if not seen_parts[owner]:
                continue
            with torch.cuda.stream(streams[owner]):
                hashes_on_device = current_layer_hashes.to(device, non_blocking=True)
                device_mask = torch.ones(len(hashes_on_device), dtype=torch.bool, device=device)
                for seen_hashes in seen_parts[owner]:
                    device_mask &= ~isin_via_searchsorted(hashes_on_device, seen_hashes)
                device_masks.append(device_mask.to(current_layer_hashes.device))
        for stream in streams:
            stream.synchronize()

        if not device_masks:
            return torch.ones(len(current_layer_hashes), dtype=torch.bool, device=current_layer_hashes.device)

        result = device_masks[0]
        for device_mask in device_masks[1:]:
            result &= device_mask
        return result

    @staticmethod
    def _merge_new_part(current_part: LayerPart, new_part: LayerPart) -> LayerPart:
        current_states, current_hashes = current_part
        new_states, new_hashes = new_part
        if len(current_hashes) == 0:
            return new_states, new_hashes
        merged_states = torch.cat([current_states, new_states], dim=0)
        merged_hashes = torch.cat([current_hashes, new_hashes], dim=0)
        merged_hashes, idx = torch.sort(merged_hashes, stable=True)
        return merged_states[idx], merged_hashes

    @classmethod
    def _bfs_layer_distributed(
        cls,
        graph: "CayleyGraph",
        layer_parts: list[LayerPart],
        seen_parts: list[list[torch.Tensor]],
        streams: list[torch.cuda.Stream],
    ) -> list[LayerPart]:
        total_size = sum(len(part_hashes) for _, part_hashes in layer_parts)
        num_batches = max(1, int(math.ceil(total_size / graph.batch_size)))
        accepted_parts = [cls._empty_part(device, graph.encoded_state_size) for device in graph.gpu_devices]
        per_gpu_batches = [list(states.tensor_split(num_batches, dim=0)) for states, _ in layer_parts]

        for batch_id in range(num_batches):
            phase1_results = [cls._empty_part(device, graph.encoded_state_size) for device in graph.gpu_devices]
            for owner, device in enumerate(graph.gpu_devices):
                chunk = per_gpu_batches[owner][batch_id]
                if len(chunk) == 0:
                    continue
                with torch.cuda.stream(streams[owner]):
                    neighbors = graph.get_neighbors(chunk)
                    phase1_results[owner] = graph.get_unique_states(neighbors)
            for stream in streams:
                stream.synchronize()

            send_states = [
                [cls._empty_part(device, graph.encoded_state_size)[0] for device in graph.gpu_devices]
                for _ in range(graph.num_gpus)
            ]
            send_hashes = [
                [torch.empty(0, dtype=torch.int64, device=device) for device in graph.gpu_devices]
                for _ in range(graph.num_gpus)
            ]
            for owner, (states, hashes) in enumerate(phase1_results):
                if len(hashes) == 0:
                    continue
                ownership = hashes % graph.num_gpus
                for target, device in enumerate(graph.gpu_devices):
                    mask = ownership == target
                    send_states[owner][target] = states[mask].to(device, non_blocking=True)
                    send_hashes[owner][target] = hashes[mask].to(device, non_blocking=True)
            torch.cuda.synchronize()

            for owner, device in enumerate(graph.gpu_devices):
                with torch.cuda.stream(streams[owner]):
                    received_states = torch.cat([send_states[source][owner] for source in range(graph.num_gpus)], dim=0)
                    received_hashes = torch.cat([send_hashes[source][owner] for source in range(graph.num_gpus)], dim=0)
                    if len(received_hashes) == 0:
                        continue

                    received_states, received_hashes = graph.get_unique_states(received_states, hashes=received_hashes)
                    for seen_hashes in seen_parts[owner]:
                        if len(received_hashes) == 0:
                            break
                        mask = ~isin_via_searchsorted(received_hashes, seen_hashes)
                        received_states, received_hashes = cls._apply_mask(received_states, received_hashes, mask)

                    if len(received_hashes) == 0:
                        continue

                    accepted_hashes = accepted_parts[owner][1]
                    if len(accepted_hashes) > 0:
                        mask = ~isin_via_searchsorted(received_hashes, accepted_hashes)
                        received_states, received_hashes = cls._apply_mask(received_states, received_hashes, mask)

                    if len(received_hashes) > 0:
                        accepted_parts[owner] = cls._merge_new_part(
                            accepted_parts[owner], (received_states, received_hashes)
                        )
            for stream in streams:
                stream.synchronize()

        return accepted_parts

    @classmethod
    def bfs(
        cls,
        graph: "CayleyGraph",
        *,
        start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**12,
        max_diameter: int = 1000000,
        return_all_hashes: bool = False,
        stop_condition: Optional[Callable[[torch.Tensor, torch.Tensor], bool]] = None,
    ) -> BfsResult:
        """Runs breadth-first search (BFS) algorithm from given ``start_states``.

        BFS visits all vertices of the graph in layers, where next layer contains vertices adjacent to previous layer
        that were not visited before. This distributed version shards the frontier and seen-state ownership across
        GPUs by hash.

        Depending on parameters below, it can be used to:
          * Get growth function (number of vertices at each BFS layer).
          * Get vertices at some first and last layers.
          * Get all vertices.

        :param graph: CayleyGraph object on which to run BFS.
        :param start_states: states on 0-th layer of BFS. Defaults to destination state of the graph.
        :param max_layer_size_to_store: maximal size of layer to store.
               If None, all layers will be stored.
               Defaults to 1000.
               First and last layers are always stored.
        :param max_layer_size_to_explore: if reaches layer of larger size, will stop the BFS.
        :param max_diameter: maximal number of BFS iterations.
        :param return_all_hashes: whether to return hashes for all vertices (uses more memory).
        :param stop_condition: function to be called after each iteration. It takes 2 tensors: latest computed layer
            and its hashes, and returns whether BFS must immediately terminate. If it returns True, the layer that was
            passed to the function will be the last returned layer in the result. This function can also be used as a
            "hook" to do some computations after BFS iteration (in which case it must always return False).
        :return: BfsResult object with requested BFS results.
        """

        if start_states is None:
            start_states = graph.central_state
        start_states = graph.encode_states(start_states)
        layer1, layer1_hashes = graph.get_unique_states(start_states)
        layer_parts = cls._partition_states(graph, layer1, layer1_hashes)
        seen_parts = [[part_hashes] if len(part_hashes) > 0 else [] for _, part_hashes in layer_parts]
        streams = [torch.cuda.Stream(device=device) for device in graph.gpu_devices]

        layer_sizes = [len(layer1)]
        layers = {0: graph.decode_states(layer1)}
        full_graph_explored = False
        all_layers_hashes = []
        max_layer_size_to_store = max_layer_size_to_store or 10**15
        last_iteration_was_distributed = False

        for layer_id in range(1, max_diameter + 1):
            total_layer_size = sum(len(part_hashes) for _, part_hashes in layer_parts)
            if total_layer_size > graph.batch_size:
                last_iteration_was_distributed = True
                if return_all_hashes:
                    all_layers_hashes.append(cls._gather_parts(graph, layer_parts)[1])
                next_parts = cls._bfs_layer_distributed(graph, layer_parts, seen_parts, streams)
                next_layer_size = sum(len(part_hashes) for _, part_hashes in next_parts)
                if next_layer_size == 0:
                    full_graph_explored = True
                    break
                if graph.verbose >= 2:
                    print(f"Layer {layer_id}: {next_layer_size} states.")
                layer_sizes.append(next_layer_size)
                if next_layer_size <= max_layer_size_to_store:
                    gathered_states, _ = cls._gather_parts(graph, next_parts)
                    layers[layer_id] = graph.decode_states(gathered_states)
                previous_parts = layer_parts
                layer_parts = next_parts
                cls._update_seen_parts(graph, seen_parts, previous_parts, next_parts)
                graph.free_memory()
                if next_layer_size >= max_layer_size_to_explore:
                    break
                if stop_condition is not None:
                    gathered_states, gathered_hashes = cls._gather_parts(graph, layer_parts)
                    if stop_condition(gathered_states, gathered_hashes):
                        break
                continue

            if last_iteration_was_distributed:
                layer1, layer1_hashes = cls._gather_parts(graph, layer_parts)
                last_iteration_was_distributed = False

            layer1_neighbors = graph.get_neighbors(layer1)
            layer1_neighbors_hashes = graph.hasher.make_hashes(layer1_neighbors)
            layer2, layer2_hashes = graph.get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)
            mask = cls._remove_seen_states(graph, seen_parts, layer2_hashes, streams)
            layer2, layer2_hashes = cls._apply_mask(layer2, layer2_hashes, mask)

            if layer2.shape[0] * layer2.shape[1] * 8 > 0.1 * graph.memory_limit_bytes:
                graph.free_memory()
            if return_all_hashes:
                all_layers_hashes.append(layer1_hashes)
            if len(layer2) == 0:
                full_graph_explored = True
                break
            if graph.verbose >= 2:
                print(f"Layer {layer_id}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            if len(layer2) <= max_layer_size_to_store:
                layers[layer_id] = graph.decode_states(layer2)

            next_parts = cls._partition_states(graph, layer2, layer2_hashes)
            cls._update_seen_parts(graph, seen_parts, layer_parts, next_parts)
            layer_parts = next_parts
            layer1, layer1_hashes = layer2, layer2_hashes
            if len(layer2) >= max_layer_size_to_explore:
                break
            if stop_condition is not None and stop_condition(layer2, layer2_hashes):
                break

        if last_iteration_was_distributed:
            layer1, layer1_hashes = cls._gather_parts(graph, layer_parts)

        if return_all_hashes and not full_graph_explored:
            all_layers_hashes.append(layer1_hashes)

        if not full_graph_explored and graph.verbose > 0:
            print("BFS stopped before graph was fully explored.")

        last_layer_id = len(layer_sizes) - 1
        if full_graph_explored and last_layer_id not in layers:
            layers[last_layer_id] = graph.decode_states(layer1)

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=layers,
            bfs_completed=full_graph_explored,
            layers_hashes=all_layers_hashes,
            edges_list_hashes=None,
            graph=graph.definition,
        )
