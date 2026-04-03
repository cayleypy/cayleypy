import math
import os
from typing import Callable, Optional, TYPE_CHECKING, Union

import numpy as np
import torch
import torch.distributed as dist

from .bfs_result import BfsResult
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


LayerPart = tuple[torch.Tensor, torch.Tensor]


class BfsDistributed:
    """Multi-GPU breadth-first search implementation.

    Public interface is preserved.

    Behavior:
      - plain python: existing single-process multi-GPU implementation;
      - torchrun with WORLD_SIZE > 1: torch.distributed multi-process implementation.
    """

    # -------------------------------------------------------------------------
    # Common small helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _empty_part(device: torch.device, state_width: int) -> LayerPart:
        return (
            torch.empty((0, state_width), dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )

    @staticmethod
    def _apply_mask(states: torch.Tensor, hashes: torch.Tensor, mask: torch.Tensor) -> LayerPart:
        return states[mask], hashes[mask]

    @staticmethod
    def _is_torchrun_env() -> bool:
        return (
            "RANK" in os.environ
            and "WORLD_SIZE" in os.environ
            and "LOCAL_RANK" in os.environ
        )

    @classmethod
    def _use_torchrun_backend(cls) -> bool:
        return cls._is_torchrun_env() and int(os.environ["WORLD_SIZE"]) > 1

    # -------------------------------------------------------------------------
    # Existing single-process helpers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # torch.distributed helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _ensure_dist_initialized(cls) -> tuple[int, int, int]:
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            return rank, world_size, local_rank

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank

    @staticmethod
    def _dist_empty_states(device: torch.device, width: int) -> torch.Tensor:
        return torch.empty((0, width), dtype=torch.int64, device=device)

    @staticmethod
    def _dist_empty_hashes(device: torch.device) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int64, device=device)

    @classmethod
    def _compact_seen_chunks(cls, chunks: list[torch.Tensor], *, threshold: int = 16) -> list[torch.Tensor]:
        non_empty = [chunk for chunk in chunks if chunk.numel() > 0]
        if len(non_empty) <= threshold:
            return non_empty
        merged = torch.unique(torch.cat(non_empty, dim=0), sorted=True)
        return [merged]

    @classmethod
    def _filter_hashes_against_chunks(cls, hashes: torch.Tensor, chunks: list[torch.Tensor]) -> torch.Tensor:
        if hashes.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=hashes.device)
        if not chunks:
            return torch.ones(hashes.shape[0], dtype=torch.bool, device=hashes.device)

        mask = torch.ones(hashes.shape[0], dtype=torch.bool, device=hashes.device)
        for seen_hashes in chunks:
            if seen_hashes.numel() == 0:
                continue
            mask &= ~isin_via_searchsorted(hashes, seen_hashes)
        return mask

    @classmethod
    def _exchange_by_owner(
        cls,
        states: torch.Tensor,
        hashes: torch.Tensor,
        world_size: int,
    ) -> LayerPart:
        """Exchange states to owner ranks via all_to_all_single.

        Each process owns hashes such that hash % world_size == rank.
        """
        device = hashes.device
        width = states.shape[1]

        if hashes.numel() > 0:
            owners = torch.remainder(hashes, world_size)
            perm = torch.argsort(owners, stable=True)
            owners = owners[perm]
            hashes = hashes[perm]
            states = states[perm]
            send_counts_hashes = torch.bincount(owners, minlength=world_size).to(torch.int64)
        else:
            send_counts_hashes = torch.zeros(world_size, dtype=torch.int64, device=device)

        recv_counts_hashes = torch.empty(world_size, dtype=torch.int64, device=device)
        dist.all_to_all_single(recv_counts_hashes, send_counts_hashes)

        recv_hashes = torch.empty(int(recv_counts_hashes.sum().item()), dtype=torch.int64, device=device)
        dist.all_to_all_single(
            recv_hashes,
            hashes,
            output_split_sizes=[int(x) for x in recv_counts_hashes.tolist()],
            input_split_sizes=[int(x) for x in send_counts_hashes.tolist()],
        )

        send_counts_states = (send_counts_hashes * width).to(torch.int64)
        recv_counts_states = (recv_counts_hashes * width).to(torch.int64)

        recv_states_flat = torch.empty(int(recv_counts_states.sum().item()), dtype=torch.int64, device=device)
        dist.all_to_all_single(
            recv_states_flat,
            states.reshape(-1),
            output_split_sizes=[int(x) for x in recv_counts_states.tolist()],
            input_split_sizes=[int(x) for x in send_counts_states.tolist()],
        )

        recv_states = recv_states_flat.view(-1, width)
        return recv_states, recv_hashes

    @classmethod
    def _gather_layer_all_ranks(
        cls,
        graph: "CayleyGraph",
        local_states: torch.Tensor,
        local_hashes: torch.Tensor,
    ) -> LayerPart:
        """Gather full layer to every rank when really needed."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            if local_hashes.numel() == 0:
                return (
                    torch.empty((0, graph.encoded_state_size), dtype=torch.int64, device=graph.device),
                    torch.empty(0, dtype=torch.int64, device=graph.device),
                )
            local_hashes, idx = torch.sort(local_hashes, stable=True)
            return local_states[idx].to(graph.device), local_hashes.to(graph.device)

        payload = (
            local_states.detach().cpu(),
            local_hashes.detach().cpu(),
        )
        gathered: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)

        states_parts = []
        hashes_parts = []
        for item in gathered:
            if item is None:
                continue
            part_states, part_hashes = item
            if part_hashes.numel() == 0:
                continue
            states_parts.append(part_states.to(graph.device))
            hashes_parts.append(part_hashes.to(graph.device))

        if not hashes_parts:
            return (
                torch.empty((0, graph.encoded_state_size), dtype=torch.int64, device=graph.device),
                torch.empty(0, dtype=torch.int64, device=graph.device),
            )

        all_states = torch.cat(states_parts, dim=0)
        all_hashes = torch.cat(hashes_parts, dim=0)
        all_hashes, idx = torch.sort(all_hashes, stable=True)
        all_states = all_states[idx]
        return all_states, all_hashes

    @staticmethod
    def _global_layer_size(local_hashes: torch.Tensor) -> int:
        size_t = torch.tensor([int(local_hashes.numel())], dtype=torch.int64, device=local_hashes.device)
        dist.all_reduce(size_t, op=dist.ReduceOp.SUM)
        return int(size_t.item())

    @staticmethod
    def _global_any_true(flag: bool, device: torch.device) -> bool:
        t = torch.tensor([1 if flag else 0], dtype=torch.int64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    @classmethod
    def _update_local_seen_chunks(
        cls,
        graph: "CayleyGraph",
        previous_hashes: torch.Tensor,
        next_hashes: torch.Tensor,
        existing_chunks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if graph.definition.generators_inverse_closed:
            chunks = [part for part in [previous_hashes, next_hashes] if part.numel() > 0]
            return cls._compact_seen_chunks(chunks)

        chunks = list(existing_chunks)
        if next_hashes.numel() > 0:
            chunks.append(next_hashes)
        return cls._compact_seen_chunks(chunks)

    @classmethod
    def _bfs_torchrun(
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
        rank, world_size, local_rank = cls._ensure_dist_initialized()
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else graph.device
        width = graph.encoded_state_size

        if start_states is None:
            start_states = graph.central_state

        if rank == 0:
            start_states_t = graph.encode_states(start_states).to(device)
            start_states_t, start_hashes_t = graph.get_unique_states(start_states_t)
        else:
            start_states_t = cls._dist_empty_states(device, width)
            start_hashes_t = cls._dist_empty_hashes(device)

        layer_states, layer_hashes = cls._exchange_by_owner(start_states_t, start_hashes_t, world_size)
        if layer_hashes.numel() > 0:
            layer_states, layer_hashes = graph.get_unique_states(layer_states, hashes=layer_hashes)

        seen_chunks = [layer_hashes] if layer_hashes.numel() > 0 else []

        first_layer_size = cls._global_layer_size(layer_hashes)
        gathered0_states, gathered0_hashes = cls._gather_layer_all_ranks(graph, layer_states, layer_hashes)

        layer_sizes = [first_layer_size]
        layers = {0: graph.decode_states(gathered0_states)}
        full_graph_explored = False
        all_layers_hashes = []

        max_layer_size_to_store = max_layer_size_to_store or 10**15

        if return_all_hashes:
            all_layers_hashes.append(gathered0_hashes)

        for layer_id in range(1, max_diameter + 1):
            frontier_size_local = layer_states.shape[0]

            accepted_state_chunks: list[torch.Tensor] = []
            accepted_hash_chunks: list[torch.Tensor] = []

            if frontier_size_local > 0:
                for start in range(0, frontier_size_local, graph.batch_size):
                    batch_states = layer_states[start : start + graph.batch_size]
                    if batch_states.numel() == 0:
                        continue

                    neighbors = graph.get_neighbors(batch_states)
                    cand_states, cand_hashes = graph.get_unique_states(neighbors)

                    recv_states, recv_hashes = cls._exchange_by_owner(cand_states, cand_hashes, world_size)
                    if recv_hashes.numel() == 0:
                        continue

                    recv_states, recv_hashes = graph.get_unique_states(recv_states, hashes=recv_hashes)

                    mask = cls._filter_hashes_against_chunks(recv_hashes, seen_chunks)
                    recv_states, recv_hashes = cls._apply_mask(recv_states, recv_hashes, mask)
                    if recv_hashes.numel() == 0:
                        continue

                    if accepted_hash_chunks:
                        tmp_chunks = cls._compact_seen_chunks(accepted_hash_chunks, threshold=8)
                        mask = cls._filter_hashes_against_chunks(recv_hashes, tmp_chunks)
                        recv_states, recv_hashes = cls._apply_mask(recv_states, recv_hashes, mask)
                        if recv_hashes.numel() == 0:
                            continue

                    accepted_state_chunks.append(recv_states)
                    accepted_hash_chunks.append(recv_hashes)

                    if len(accepted_hash_chunks) > 16:
                        merged_states = torch.cat(accepted_state_chunks, dim=0)
                        merged_hashes = torch.cat(accepted_hash_chunks, dim=0)
                        merged_states, merged_hashes = graph.get_unique_states(merged_states, hashes=merged_hashes)
                        accepted_state_chunks = [merged_states]
                        accepted_hash_chunks = [merged_hashes]

            if accepted_hash_chunks:
                next_states = torch.cat(accepted_state_chunks, dim=0)
                next_hashes = torch.cat(accepted_hash_chunks, dim=0)
                next_states, next_hashes = graph.get_unique_states(next_states, hashes=next_hashes)

                mask = cls._filter_hashes_against_chunks(next_hashes, seen_chunks)
                next_states, next_hashes = cls._apply_mask(next_states, next_hashes, mask)
            else:
                next_states = cls._dist_empty_states(device, width)
                next_hashes = cls._dist_empty_hashes(device)

            next_layer_size = cls._global_layer_size(next_hashes)

            if next_layer_size == 0:
                full_graph_explored = True
                break

            if graph.verbose >= 2 and rank == 0:
                print(f"Layer {layer_id}: {next_layer_size} states.")

            layer_sizes.append(next_layer_size)

            need_gather = (
                next_layer_size <= max_layer_size_to_store
                or return_all_hashes
                or stop_condition is not None
            )

            gathered_states = None
            gathered_hashes = None
            if need_gather:
                gathered_states, gathered_hashes = cls._gather_layer_all_ranks(graph, next_states, next_hashes)

            if next_layer_size <= max_layer_size_to_store:
                assert gathered_states is not None
                layers[layer_id] = graph.decode_states(gathered_states)

            if return_all_hashes:
                assert gathered_hashes is not None
                all_layers_hashes.append(gathered_hashes)

            previous_hashes = layer_hashes
            layer_states, layer_hashes = next_states, next_hashes
            seen_chunks = cls._update_local_seen_chunks(graph, previous_hashes, layer_hashes, seen_chunks)

            if layer_hashes.shape[0] * width * 8 > 0.1 * graph.memory_limit_bytes:
                graph.free_memory()

            if next_layer_size >= max_layer_size_to_explore:
                break

            if stop_condition is not None:
                assert gathered_states is not None and gathered_hashes is not None
                stop_now_local = bool(stop_condition(gathered_states, gathered_hashes))
                stop_now = cls._global_any_true(stop_now_local, device)
                if stop_now:
                    break

        if not full_graph_explored and graph.verbose > 0 and rank == 0:
            print("BFS stopped before graph was fully explored.")

        last_layer_id = len(layer_sizes) - 1
        if full_graph_explored and last_layer_id not in layers:
            gathered_last_states, _ = cls._gather_layer_all_ranks(graph, layer_states, layer_hashes)
            layers[last_layer_id] = graph.decode_states(gathered_last_states)

        dist.barrier()

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=layers,
            bfs_completed=full_graph_explored,
            layers_hashes=all_layers_hashes,
            edges_list_hashes=None,
            graph=graph.definition,
        )

    # -------------------------------------------------------------------------
    # Existing implementation preserved as single-process path
    # -------------------------------------------------------------------------

    @classmethod
    def _bfs_single_process(
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

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

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

        Behavior is selected automatically:
          * plain python -> existing single-process implementation;
          * torchrun with WORLD_SIZE > 1 -> torch.distributed implementation.
        """
        if cls._use_torchrun_backend():
            return cls._bfs_torchrun(
                graph,
                start_states=start_states,
                max_layer_size_to_store=max_layer_size_to_store,
                max_layer_size_to_explore=max_layer_size_to_explore,
                max_diameter=max_diameter,
                return_all_hashes=return_all_hashes,
                stop_condition=stop_condition,
            )

        return cls._bfs_single_process(
            graph,
            start_states=start_states,
            max_layer_size_to_store=max_layer_size_to_store,
            max_layer_size_to_explore=max_layer_size_to_explore,
            max_diameter=max_diameter,
            return_all_hashes=return_all_hashes,
            stop_condition=stop_condition,
        )