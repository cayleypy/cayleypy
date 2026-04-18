"""Torchrun-based beam search algorithms for Cayley graphs."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from .beam_search import BeamSearchAlgorithm
from .beam_search_result import BeamSearchResult
from ..cayley_graph_def import AnyStateType
from ..predictor import Predictor

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


STOP_CONTINUE = 0
STOP_EMPTY = 1
STOP_FOUND = 2


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def _use_distributed_backend() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1
    return _is_torchrun_env() and int(os.environ["WORLD_SIZE"]) > 1


def _select_rank_device(graph: "CayleyGraph", rank: int) -> torch.device:
    if graph.device.type == "cuda" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return graph.device


def _ensure_distributed_context(graph: "CayleyGraph") -> tuple[int, int, torch.device]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, _select_rank_device(graph, rank)

    if not _is_torchrun_env() or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 1, graph.device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = _select_rank_device(graph, rank)
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, device


def _encode_states_to_device(graph: "CayleyGraph", states: AnyStateType, device: torch.device) -> torch.Tensor:
    states_t = torch.as_tensor(states, device=device, dtype=torch.int64)
    states_t = states_t.reshape((-1, graph.definition.state_size))
    if graph.string_encoder is not None:
        return graph.string_encoder.encode(states_t)
    return states_t


def _empty_states(device: torch.device, width: int) -> torch.Tensor:
    return torch.empty((0, width), dtype=torch.int64, device=device)


def _empty_scores(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=device)


def _normalize_states(states: torch.Tensor, width: int) -> torch.Tensor:
    if states.dim() == 1:
        return states.reshape(1, width)
    if states.dim() > 2:
        return states.flatten(end_dim=1)
    return states


def _as_score_tensor(scores: object, *, device: torch.device, expected_size: int) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        scores_t = scores.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        scores_t = torch.as_tensor(scores, dtype=torch.float32, device=device).reshape(-1)
    if scores_t.numel() != expected_size:
        raise ValueError(f"predictor returned {scores_t.numel()} scores for {expected_size} states")
    return torch.nan_to_num(scores_t, nan=float("inf"), posinf=float("inf"), neginf=float("-inf"))


def _score_states(
    graph: "CayleyGraph",
    states: torch.Tensor,
    predictor: Optional[Predictor],
    device: torch.device,
) -> torch.Tensor:
    if states.numel() == 0:
        return _empty_scores(device)
    if predictor is None:
        decoded_states = graph.decode_states(states)
        central_state = graph.central_state.to(decoded_states.device)
        scores = (decoded_states != central_state).reshape(decoded_states.shape[0], -1).sum(dim=1)
        return _as_score_tensor(scores, device=device, expected_size=states.shape[0])
    scores = predictor(graph.decode_states(states))
    return _as_score_tensor(scores, device=device, expected_size=states.shape[0])


def _compute_per_rank_beam(beam_width: int, world_size: int, rank: int) -> int:
    base = beam_width // world_size
    rem = beam_width % world_size
    return base + (1 if rank < rem else 0)


def _compute_owner_pre_k(
    beam_width: int,
    world_size: int,
    owner_rank: int,
    oversubscription_factor: int,
) -> int:
    owner_target = _compute_per_rank_beam(beam_width, world_size, owner_rank)
    if owner_target == 0:
        return 0
    return max(owner_target, int(owner_target * oversubscription_factor))


def _topk_by_score(states: torch.Tensor, scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k <= 0:
        return _empty_states(states.device, states.shape[1]), _empty_scores(scores.device)
    if states.shape[0] <= k:
        return states.contiguous(), scores.contiguous()
    idx = torch.argsort(scores, stable=True)[:k]
    return states[idx].contiguous(), scores[idx].contiguous()


def _deduplicate_keep_best_score(
    graph: "CayleyGraph",
    states: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if states.numel() == 0:
        return states, graph.hasher.make_hashes(states), scores

    hashes = graph.hasher.make_hashes(states)
    score_order = torch.argsort(scores, stable=True)
    states = states[score_order]
    scores = scores[score_order]
    hashes = hashes[score_order]

    hash_order = torch.argsort(hashes, stable=True)
    states = states[hash_order]
    scores = scores[hash_order]
    hashes = hashes[hash_order]

    keep = torch.ones(hashes.shape[0], dtype=torch.bool, device=hashes.device)
    if hashes.shape[0] > 1:
        keep[1:] = hashes[1:] != hashes[:-1]
    return states[keep].contiguous(), hashes[keep].contiguous(), scores[keep].contiguous()


def _filter_history(
    states: torch.Tensor,
    hashes: torch.Tensor,
    scores: torch.Tensor,
    history_hashes: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hashes.numel() == 0 or not history_hashes:
        return states, hashes, scores

    mask = torch.ones(hashes.shape[0], dtype=torch.bool, device=hashes.device)
    for old_hashes in history_hashes:
        if old_hashes.numel() > 0:
            mask &= ~torch.isin(hashes, old_hashes, assume_unique=False)
    return states[mask].contiguous(), hashes[mask].contiguous(), scores[mask].contiguous()


def _update_history(
    history_hashes: list[torch.Tensor],
    next_hashes: torch.Tensor,
    history_depth: int,
) -> list[torch.Tensor]:
    if history_depth <= 0:
        return []
    updated = list(history_hashes)
    updated.append(next_hashes.detach())
    return updated[-history_depth:]


def _reduce_step_status(
    found_local: bool,
    nonempty_local: bool,
    local_best: float,
    device: torch.device,
) -> tuple[int, float]:
    status = torch.tensor([int(found_local), int(nonempty_local)], dtype=torch.int64, device=device)
    best_score = torch.tensor([local_best], dtype=torch.float32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    dist.all_reduce(best_score, op=dist.ReduceOp.MIN)

    stop_local = STOP_CONTINUE
    if int(status[0].item()) > 0:
        stop_local = STOP_FOUND
    elif int(status[1].item()) == 0:
        stop_local = STOP_EMPTY

    stop_code = torch.tensor([stop_local], dtype=torch.int64, device=device)
    dist.all_reduce(stop_code, op=dist.ReduceOp.MAX)
    return int(stop_code.item()), float(best_score.item())


def _fallback_advanced(
    graph: "CayleyGraph",
    *,
    start_state: AnyStateType,
    destination_state: Optional[AnyStateType],
    beam_width: int,
    max_steps: int,
    history_depth: int,
    predictor: Optional[Predictor],
    verbose: int,
) -> BeamSearchResult:
    return BeamSearchAlgorithm(graph).search_advanced(
        start_state=start_state,
        destination_state=destination_state,
        beam_width=beam_width,
        max_steps=max_steps,
        history_depth=history_depth,
        predictor=predictor,
        verbose=verbose,
    )


def search_multigpu(
    graph: "CayleyGraph",
    *,
    start_state: AnyStateType,
    destination_state: Optional[AnyStateType] = None,
    beam_width: int = 1000,
    max_steps: int = 1000,
    history_depth: int = 0,
    predictor: Optional[Predictor] = None,
    strategy: str = "owner_partitioned",
    oversubscription_factor: int = 2,
    verbose: int = 0,
) -> BeamSearchResult:
    """Run beam search with a torchrun distributed path when WORLD_SIZE > 1.

    Single-process execution delegates to ``BeamSearchAlgorithm.search_advanced``.
    Under torchrun, the default strategy is owner-partitioned routing, which avoids
    gathering all candidate states on every rank.
    """
    if not _use_distributed_backend():
        return _fallback_advanced(
            graph,
            start_state=start_state,
            destination_state=destination_state,
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            predictor=predictor,
            verbose=verbose,
        )

    if strategy == "owner_partitioned":
        return search_multigpu_owner_partitioned(
            graph,
            start_state=start_state,
            destination_state=destination_state,
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            predictor=predictor,
            oversubscription_factor=oversubscription_factor,
            verbose=verbose,
        )
    if strategy == "all_gather":
        return search_multigpu_all_gather(
            graph,
            start_state=start_state,
            destination_state=destination_state,
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            predictor=predictor,
            oversubscription_factor=oversubscription_factor,
            verbose=verbose,
        )
    raise ValueError(f"Unknown multi-GPU beam search strategy: {strategy}")


def search_multigpu_owner_partitioned(
    graph: "CayleyGraph",
    *,
    start_state: AnyStateType,
    destination_state: Optional[AnyStateType] = None,
    beam_width: int = 1000,
    max_steps: int = 1000,
    history_depth: int = 0,
    predictor: Optional[Predictor] = None,
    oversubscription_factor: int = 2,
    verbose: int = 0,
) -> BeamSearchResult:
    """Scalable torchrun beam search with owner-based state partitioning."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if oversubscription_factor < 1:
        raise ValueError("oversubscription_factor must be >= 1")

    rank, world_size, device = _ensure_distributed_context(graph)
    if world_size <= 1:
        return _fallback_advanced(
            graph,
            start_state=start_state,
            destination_state=destination_state,
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            predictor=predictor,
            verbose=verbose,
        )

    width = graph.encoded_state_size
    start_encoded = _encode_states_to_device(graph, start_state, device)
    dest_state = destination_state if destination_state is not None else graph.central_state
    dest_encoded = _encode_states_to_device(graph, dest_state, device)

    start_found_local = bool(torch.any(torch.all(start_encoded == dest_encoded, dim=1)).item())
    start_found = torch.tensor([int(start_found_local)], dtype=torch.int64, device=device)
    dist.all_reduce(start_found, op=dist.ReduceOp.MAX)
    if int(start_found.item()) > 0:
        return BeamSearchResult(True, 0, [], {}, graph.definition)

    start_states, start_hashes = graph.get_unique_states(start_encoded)
    start_owners = torch.remainder(start_hashes, world_size)
    start_mask = start_owners == rank
    local_beam_states = start_states[start_mask].contiguous()
    local_beam_hashes = start_hashes[start_mask].contiguous()
    history_hashes = [local_beam_hashes.detach()] if history_depth > 0 and local_beam_hashes.numel() > 0 else []
    debug_scores: dict[int, float] = {}

    for step in range(1, max_steps + 1):
        if local_beam_states.numel() == 0:
            local_candidates = _empty_states(device, width)
            candidate_scores = _empty_scores(device)
            candidate_hashes = torch.empty((0,), dtype=torch.int64, device=device)
        else:
            local_candidates = _normalize_states(graph.get_neighbors(local_beam_states), width)
            local_candidates, candidate_hashes = graph.get_unique_states(local_candidates)
            candidate_scores = _score_states(graph, local_candidates, predictor, device)

        found_local = False
        if local_candidates.numel() > 0:
            found_local = bool(torch.any(torch.all(local_candidates == dest_encoded, dim=1)).item())

        owners = torch.remainder(candidate_hashes, world_size) if candidate_hashes.numel() > 0 else candidate_hashes
        send_states_parts: list[torch.Tensor] = []
        send_scores_parts: list[torch.Tensor] = []
        send_counts: list[int] = []

        for owner_rank in range(world_size):
            owner_budget = _compute_owner_pre_k(beam_width, world_size, owner_rank, oversubscription_factor)
            if owners.numel() == 0 or owner_budget == 0:
                dest_states = _empty_states(device, width)
                dest_scores = _empty_scores(device)
            else:
                mask = owners == owner_rank
                dest_states = local_candidates[mask]
                dest_scores = candidate_scores[mask]
                dest_states, dest_scores = _topk_by_score(dest_states, dest_scores, owner_budget)
            send_states_parts.append(dest_states)
            send_scores_parts.append(dest_scores)
            send_counts.append(int(dest_states.shape[0]))

        send_states = torch.cat(send_states_parts, dim=0).contiguous()
        send_scores = torch.cat(send_scores_parts, dim=0).contiguous()
        send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=device)

        gathered_counts = torch.empty((world_size * world_size,), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(gathered_counts, send_counts_t)
        count_matrix = gathered_counts.view(world_size, world_size)
        recv_counts_t = count_matrix[:, rank].contiguous()
        recv_counts = [int(x) for x in recv_counts_t.tolist()]
        total_recv = int(recv_counts_t.sum().item())

        recv_states = torch.empty((total_recv, width), dtype=torch.int64, device=device)
        recv_scores = torch.empty((total_recv,), dtype=torch.float32, device=device)
        dist.all_to_all_single(
            recv_states,
            send_states,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        dist.all_to_all_single(
            recv_scores,
            send_scores,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )

        unique_states, unique_hashes, unique_scores = _deduplicate_keep_best_score(graph, recv_states, recv_scores)
        if history_depth > 0:
            unique_states, unique_hashes, unique_scores = _filter_history(
                unique_states,
                unique_hashes,
                unique_scores,
                history_hashes,
            )

        my_target = _compute_per_rank_beam(beam_width, world_size, rank)
        local_beam_states, next_scores = _topk_by_score(unique_states, unique_scores, my_target)
        if local_beam_states.numel() == 0:
            local_beam_hashes = torch.empty((0,), dtype=torch.int64, device=device)
        else:
            local_beam_hashes = graph.hasher.make_hashes(local_beam_states)
        history_hashes = _update_history(history_hashes, local_beam_hashes, history_depth)

        local_best = float(next_scores.min().item()) if next_scores.numel() > 0 else float("inf")
        stop_code, best_score = _reduce_step_status(
            found_local,
            local_beam_states.shape[0] > 0,
            local_best,
            device,
        )
        if math.isfinite(best_score):
            debug_scores[step] = best_score

        if verbose >= 2:
            global_count = torch.tensor([int(local_beam_states.shape[0])], dtype=torch.int64, device=device)
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(f"Step {step}: beam={int(global_count.item())}, best_score={best_score:.6f}.")

        if stop_code == STOP_FOUND:
            if verbose >= 1 and rank == 0:
                print(f"Destination found at step {step}.")
            return BeamSearchResult(True, step, None, debug_scores, graph.definition)
        if stop_code == STOP_EMPTY:
            if verbose >= 1 and rank == 0:
                print(f"No beam candidates remain at step {step}.")
            return BeamSearchResult(False, step, None, debug_scores, graph.definition)

    if verbose >= 1 and rank == 0:
        print(f"Beam search did not converge within {max_steps} steps.")
    return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)


def search_multigpu_all_gather(
    graph: "CayleyGraph",
    *,
    start_state: AnyStateType,
    destination_state: Optional[AnyStateType] = None,
    beam_width: int = 1000,
    max_steps: int = 1000,
    history_depth: int = 0,
    predictor: Optional[Predictor] = None,
    oversubscription_factor: int = 2,
    verbose: int = 0,
) -> BeamSearchResult:
    """Torchrun beam search that gathers bounded candidates on every rank."""
    rank, world_size, device = _ensure_distributed_context(graph)
    if world_size <= 1:
        return _fallback_advanced(
            graph,
            start_state=start_state,
            destination_state=destination_state,
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            predictor=predictor,
            verbose=verbose,
        )

    width = graph.encoded_state_size
    start_encoded = _encode_states_to_device(graph, start_state, device)
    dest_state = destination_state if destination_state is not None else graph.central_state
    dest_encoded = _encode_states_to_device(graph, dest_state, device)
    start_found_local = bool(torch.any(torch.all(start_encoded == dest_encoded, dim=1)).item())
    start_found = torch.tensor([int(start_found_local)], dtype=torch.int64, device=device)
    dist.all_reduce(start_found, op=dist.ReduceOp.MAX)
    if int(start_found.item()) > 0:
        return BeamSearchResult(True, 0, [], {}, graph.definition)

    global_start, global_start_hashes = graph.get_unique_states(start_encoded)
    local_beam_states = global_start[rank::world_size].contiguous()
    history_hashes = [global_start_hashes.detach()] if history_depth > 0 else []
    debug_scores: dict[int, float] = {}
    gather_limit = max(1, min(beam_width, math.ceil(beam_width / world_size) * oversubscription_factor))

    for step in range(1, max_steps + 1):
        if local_beam_states.numel() == 0:
            local_candidates = _empty_states(device, width)
            local_scores = _empty_scores(device)
        else:
            local_candidates = _normalize_states(graph.get_neighbors(local_beam_states), width)
            local_candidates, _ = graph.get_unique_states(local_candidates)
            local_scores = _score_states(graph, local_candidates, predictor, device)

        found_local = False
        if local_candidates.numel() > 0:
            found_local = bool(torch.any(torch.all(local_candidates == dest_encoded, dim=1)).item())

        local_candidates, local_scores = _topk_by_score(local_candidates, local_scores, gather_limit)
        padded_states = torch.zeros((gather_limit, width), dtype=torch.int64, device=device)
        padded_scores = torch.full((gather_limit,), float("inf"), dtype=torch.float32, device=device)
        local_count = local_candidates.shape[0]
        if local_count > 0:
            padded_states[:local_count] = local_candidates
            padded_scores[:local_count] = local_scores

        gathered_states = torch.empty((world_size * gather_limit, width), dtype=torch.int64, device=device)
        gathered_scores = torch.empty((world_size * gather_limit,), dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered_states, padded_states)
        dist.all_gather_into_tensor(gathered_scores, padded_scores)

        valid_mask = gathered_scores != float("inf")
        gathered_states = gathered_states[valid_mask]
        gathered_scores = gathered_scores[valid_mask]
        unique_states, unique_hashes, unique_scores = _deduplicate_keep_best_score(
            graph,
            gathered_states,
            gathered_scores,
        )
        if history_depth > 0:
            unique_states, unique_hashes, unique_scores = _filter_history(
                unique_states,
                unique_hashes,
                unique_scores,
                history_hashes,
            )

        global_states, global_scores = _topk_by_score(unique_states, unique_scores, beam_width)
        global_hashes = graph.hasher.make_hashes(global_states) if global_states.numel() > 0 else unique_hashes
        history_hashes = _update_history(history_hashes, global_hashes, history_depth)
        local_beam_states = global_states[rank::world_size].contiguous()
        local_scores_next = global_scores[rank::world_size].contiguous()

        local_best = float(local_scores_next.min().item()) if local_scores_next.numel() > 0 else float("inf")
        stop_code, best_score = _reduce_step_status(
            found_local,
            global_states.shape[0] > 0,
            local_best,
            device,
        )
        if math.isfinite(best_score):
            debug_scores[step] = best_score

        if verbose >= 2 and rank == 0:
            print(f"Step {step}: beam={global_states.shape[0]}, best_score={best_score:.6f}.")

        if stop_code == STOP_FOUND:
            if verbose >= 1 and rank == 0:
                print(f"Destination found at step {step}.")
            return BeamSearchResult(True, step, None, debug_scores, graph.definition)
        if stop_code == STOP_EMPTY:
            if verbose >= 1 and rank == 0:
                print(f"No beam candidates remain at step {step}.")
            return BeamSearchResult(False, step, None, debug_scores, graph.definition)

    if verbose >= 1 and rank == 0:
        print(f"Beam search did not converge within {max_steps} steps.")
    return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)


__all__ = [
    "search_multigpu",
    "search_multigpu_all_gather",
    "search_multigpu_owner_partitioned",
]
