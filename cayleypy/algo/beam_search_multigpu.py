"""Torchrun-based beam search algorithms for Cayley graphs.

CUDA-focused implementation notes:

- `search_multigpu_owner_partitioned` is the only distributed production path.
- Owner-partitioned search uses streaming expansion, chunked predictor scoring,
  bounded per-owner GPU candidate accumulation, partial top-k, and owner-local history.
- CPU / Gloo distributed execution is intentionally not supported here.
- Multi-GPU scaling target:
    - memory scales approximately with world size because each rank stores only
      owner-local beam/history plus bounded pre-pruned owner-local candidate buffers;
    - larger world size therefore permits substantially larger global beam width.

Memory-oriented design changes:

- repeated incremental `torch.cat(...)` inside the chunk loop is removed;
- predictor scoring uses a single preallocated output tensor instead of list+cat;
- top-k selection no longer performs an unnecessary second full sort;
- send/recv tensors are reused via a simple workspace allocator;
- dedup performs a single sort by hash and only gathers surviving rows;
- large `.contiguous()` copies are avoided unless actually useful;
- adaptive chunk sizing uses a more realistic temporary-memory model;
- if graph exposes `get_neighbors_chunked(...)`, neighbor expansion is streamed
  in generator chunks instead of materializing the full neighborhood tensor.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

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

_DISTRIBUTED_CONTEXT_OWNS_PG = False

# Signed int64 equivalents of 64-bit MurmurHash3 fmix64 constants.
_OWNER_MIX_C1 = -49064778989728563
_OWNER_MIX_C2 = -4265267296055464877
_OWNER_MIX_TENSORS: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def _use_distributed_backend() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1
    return _is_torchrun_env() and int(os.environ["WORLD_SIZE"]) > 1


def _select_rank_device(graph: "CayleyGraph", rank: int) -> torch.device:
    if graph.device.type == "cuda" and torch.cuda.is_available():
        device_count = max(1, torch.cuda.device_count())
        local_rank = int(os.environ.get("LOCAL_RANK", rank % device_count))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return graph.device


def _ensure_distributed_context(graph: "CayleyGraph") -> tuple[int, int, torch.device]:
    global _DISTRIBUTED_CONTEXT_OWNS_PG

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = _select_rank_device(graph, rank)
        if device.type != "cuda":
            raise RuntimeError("Distributed beam search in this module is CUDA/NCCL-only.")
        return rank, world_size, device

    if not _is_torchrun_env() or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 1, graph.device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = _select_rank_device(graph, rank)
    if device.type != "cuda":
        raise RuntimeError("Distributed beam search in this module is CUDA/NCCL-only.")

    dist.init_process_group(backend="nccl", init_method="env://")
    _DISTRIBUTED_CONTEXT_OWNS_PG = True
    return rank, world_size, device


def _cleanup_distributed_context() -> None:
    global _DISTRIBUTED_CONTEXT_OWNS_PG

    if not _DISTRIBUTED_CONTEXT_OWNS_PG:
        return
    if not dist.is_available() or not dist.is_initialized():
        _DISTRIBUTED_CONTEXT_OWNS_PG = False
        return

    try:
        dist.destroy_process_group()
    finally:
        _DISTRIBUTED_CONTEXT_OWNS_PG = False


def _encode_states_to_device(graph: "CayleyGraph", states: AnyStateType, device: torch.device) -> torch.Tensor:
    states_t = torch.as_tensor(states, device=device, dtype=torch.int64)
    states_t = states_t.reshape((-1, graph.definition.state_size))
    if graph.string_encoder is not None:
        encoded = graph.string_encoder.encode(states_t)
        return encoded.reshape((-1, graph.encoded_state_size))
    return states_t


def _empty_states(device: torch.device, width: int, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    return torch.empty((0, width), dtype=dtype, device=device)


def _empty_scores(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=device)


def _empty_hashes(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int64, device=device)


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

    return torch.nan_to_num(
        scores_t,
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("-inf"),
    )


def _predictor_score_encoded_if_available(
    predictor: Optional[Predictor],
    states: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if predictor is None:
        return None

    score_encoded = getattr(predictor, "score_encoded", None)
    if callable(score_encoded):
        scores = score_encoded(states)
        return _as_score_tensor(scores, device=device, expected_size=states.shape[0])

    score_states = getattr(predictor, "score_states", None)
    if callable(score_states):
        scores = score_states(states)
        return _as_score_tensor(scores, device=device, expected_size=states.shape[0])

    return None


def _is_cuda_oom(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _cleanup_after_oom(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


@dataclass
class _SearchWorkspace:
    device: torch.device
    state_dtype: torch.dtype
    width: int

    _send_states: Optional[torch.Tensor] = None
    _send_scores: Optional[torch.Tensor] = None
    _recv_states: Optional[torch.Tensor] = None
    _recv_scores: Optional[torch.Tensor] = None
    _score_out: Optional[torch.Tensor] = None

    def _ensure_state_buffer(self, attr: str, rows: int) -> torch.Tensor:
        rows = max(0, int(rows))
        buf = getattr(self, attr)
        if buf is None or buf.shape[0] < rows:
            new_rows = max(rows, int(math.ceil(max(1, rows) * 1.25)))
            buf = torch.empty((new_rows, self.width), dtype=self.state_dtype, device=self.device)
            setattr(self, attr, buf)
        return buf[:rows]

    def _ensure_score_buffer(self, attr: str, rows: int) -> torch.Tensor:
        rows = max(0, int(rows))
        buf = getattr(self, attr)
        if buf is None or buf.shape[0] < rows:
            new_rows = max(rows, int(math.ceil(max(1, rows) * 1.25)))
            buf = torch.empty((new_rows,), dtype=torch.float32, device=self.device)
            setattr(self, attr, buf)
        return buf[:rows]

    def get_send_states(self, rows: int) -> torch.Tensor:
        return self._ensure_state_buffer("_send_states", rows)

    def get_send_scores(self, rows: int) -> torch.Tensor:
        return self._ensure_score_buffer("_send_scores", rows)

    def get_recv_states(self, rows: int) -> torch.Tensor:
        return self._ensure_state_buffer("_recv_states", rows)

    def get_recv_scores(self, rows: int) -> torch.Tensor:
        return self._ensure_score_buffer("_recv_scores", rows)

    def get_score_out(self, rows: int) -> torch.Tensor:
        return self._ensure_score_buffer("_score_out", rows)


def _score_states_single_batch(
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

    direct_scores = _predictor_score_encoded_if_available(predictor, states, device)
    if direct_scores is not None:
        return direct_scores

    decoded_scores = predictor(graph.decode_states(states))
    return _as_score_tensor(decoded_scores, device=device, expected_size=states.shape[0])


def _safe_score_states(
    graph: "CayleyGraph",
    states: torch.Tensor,
    predictor: Optional[Predictor],
    device: torch.device,
    *,
    predictor_batch_size: int,
    workspace: _SearchWorkspace,
) -> torch.Tensor:
    if states.numel() == 0:
        return _empty_scores(device)

    n = states.shape[0]
    if predictor_batch_size <= 0 or n <= predictor_batch_size:
        return _score_states_single_batch(graph, states, predictor, device)

    out = workspace.get_score_out(n)
    offset = 0
    for batch in states.split(predictor_batch_size, dim=0):
        batch_scores = _score_states_single_batch(graph, batch, predictor, device)
        batch_n = batch.shape[0]
        out[offset : offset + batch_n].copy_(batch_scores)
        offset += batch_n
    return out[:n]


def _score_states_oom_safe(
    graph: "CayleyGraph",
    states: torch.Tensor,
    predictor: Optional[Predictor],
    device: torch.device,
    *,
    predictor_batch_size: int,
    workspace: _SearchWorkspace,
) -> torch.Tensor:
    if states.numel() == 0:
        return _empty_scores(device)

    if predictor_batch_size <= 0:
        try:
            return _safe_score_states(
                graph,
                states,
                predictor,
                device,
                predictor_batch_size=predictor_batch_size,
                workspace=workspace,
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if states.shape[0] <= 1:
                raise
            _cleanup_after_oom(device)
            predictor_batch_size = max(1, states.shape[0] // 2)

    current_batch = min(states.shape[0], predictor_batch_size if predictor_batch_size > 0 else states.shape[0])
    current_batch = max(1, current_batch)

    while True:
        try:
            return _safe_score_states(
                graph,
                states,
                predictor,
                device,
                predictor_batch_size=current_batch,
                workspace=workspace,
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if current_batch <= 1:
                raise
            _cleanup_after_oom(device)
            current_batch = max(1, current_batch // 2)


def _compute_per_rank_beam(beam_width: int, world_size: int, rank: int) -> int:
    base = beam_width // world_size
    rem = beam_width % world_size
    return base + (1 if rank < rem else 0)


def _compute_owner_pre_k(
    beam_width: int,
    world_size: int,
    owner_rank: int,
    oversubscription_factor: float,
) -> int:
    owner_target = _compute_per_rank_beam(beam_width, world_size, owner_rank)
    if owner_target == 0:
        return 0
    return max(owner_target, int(math.ceil(owner_target * oversubscription_factor)))


def _topk_by_score(states: torch.Tensor, scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k <= 0:
        return _empty_states(states.device, states.shape[1], states.dtype), _empty_scores(scores.device)

    n = states.shape[0]
    if n == 0:
        return _empty_states(states.device, states.shape[1], states.dtype), _empty_scores(scores.device)

    if n <= k:
        return states, scores

    _, idx = torch.topk(scores, k, largest=False, sorted=False)
    return states[idx], scores[idx]


def _topk_scores_only(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return _empty_scores(scores.device)
    if scores.numel() <= k:
        return scores
    return torch.topk(scores, k, largest=False, sorted=False).values


def _deduplicate_keep_best_score_fallback(
    order: torch.Tensor,
    sorted_hashes: torch.Tensor,
    sorted_scores: torch.Tensor,
) -> torch.Tensor:
    n = sorted_hashes.shape[0]
    if n == 0:
        return order[:0]

    keep_positions: list[int] = []
    start = 0
    while start < n:
        end = start + 1
        hash_value = sorted_hashes[start]
        while end < n and sorted_hashes[end] == hash_value:
            end += 1
        local_scores = sorted_scores[start:end]
        rel = int(torch.argmin(local_scores).item())
        keep_positions.append(start + rel)
        start = end

    keep_pos_t = torch.tensor(keep_positions, dtype=torch.int64, device=order.device)
    return order[keep_pos_t]


def _deduplicate_keep_best_score(
    graph: "CayleyGraph",
    states: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if states.numel() == 0:
        return states, graph.hasher.make_hashes(states), scores

    hashes = graph.hasher.make_hashes(states)
    order = torch.argsort(hashes)
    sorted_hashes = hashes[order]
    sorted_scores = scores[order]

    if sorted_hashes.shape[0] <= 1:
        keep_order = order
    else:
        try:
            unique_hashes, inverse = torch.unique_consecutive(sorted_hashes, return_inverse=True)
            min_scores = torch.full(
                (unique_hashes.shape[0],),
                float("inf"),
                dtype=sorted_scores.dtype,
                device=sorted_scores.device,
            )
            min_scores.scatter_reduce_(0, inverse, sorted_scores, reduce="amin", include_self=True)

            is_best = sorted_scores == min_scores[inverse]
            best_positions = torch.nonzero(is_best, as_tuple=False).reshape(-1)

            best_group_ids = inverse[best_positions]
            first_best = torch.ones(best_positions.shape[0], dtype=torch.bool, device=best_positions.device)
            if best_positions.shape[0] > 1:
                first_best[1:] = best_group_ids[1:] != best_group_ids[:-1]

            keep_order = order[best_positions[first_best]]
        except Exception:
            keep_order = _deduplicate_keep_best_score_fallback(order, sorted_hashes, sorted_scores)

    kept_states = states[keep_order]
    kept_hashes = hashes[keep_order]
    kept_scores = scores[keep_order]
    return kept_states, kept_hashes, kept_scores


def _sorted_unique_hashes(hashes: torch.Tensor) -> torch.Tensor:
    if hashes.numel() == 0:
        return hashes
    sorted_hashes = torch.sort(hashes).values
    keep = torch.ones(sorted_hashes.shape[0], dtype=torch.bool, device=sorted_hashes.device)
    if sorted_hashes.shape[0] > 1:
        keep[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    return sorted_hashes[keep]


def _contains_sorted_hashes(sorted_haystack: torch.Tensor, needles: torch.Tensor) -> torch.Tensor:
    if needles.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=needles.device)
    if sorted_haystack.numel() == 0:
        return torch.zeros(needles.shape[0], dtype=torch.bool, device=needles.device)

    pos = torch.searchsorted(sorted_haystack, needles)
    pos_clamped = torch.clamp(pos, max=sorted_haystack.shape[0] - 1)
    return sorted_haystack[pos_clamped] == needles


def _filter_hashes_against_history(
    hashes: torch.Tensor,
    history_hashes: list[torch.Tensor],
) -> torch.Tensor:
    if hashes.numel() == 0 or not history_hashes:
        return torch.ones(hashes.shape[0], dtype=torch.bool, device=hashes.device)

    mask = torch.ones(hashes.shape[0], dtype=torch.bool, device=hashes.device)
    for old_hashes in history_hashes:
        if old_hashes.numel() > 0:
            mask &= ~_contains_sorted_hashes(old_hashes, hashes)
            if not torch.any(mask):
                break
    return mask


def _filter_history(
    states: torch.Tensor,
    hashes: torch.Tensor,
    scores: torch.Tensor,
    history_hashes: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hashes.numel() == 0 or not history_hashes:
        return states, hashes, scores

    mask = _filter_hashes_against_history(hashes, history_hashes)
    if torch.all(mask):
        return states, hashes, scores
    return states[mask], hashes[mask], scores[mask]


def _update_history(
    history_hashes: list[torch.Tensor],
    next_hashes: torch.Tensor,
    history_depth: int,
) -> list[torch.Tensor]:
    if history_depth <= 0:
        return []

    sorted_next = _sorted_unique_hashes(next_hashes.detach())
    updated = list(history_hashes)
    updated.append(sorted_next)
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


def _get_owner_mix_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, torch.int64)
    cached = _OWNER_MIX_TENSORS.get(key)
    if cached is None:
        cached = (
            torch.tensor(_OWNER_MIX_C1, dtype=torch.int64, device=device),
            torch.tensor(_OWNER_MIX_C2, dtype=torch.int64, device=device),
        )
        _OWNER_MIX_TENSORS[key] = cached
    return cached


def _owner_mix_hashes(hashes: torch.Tensor) -> torch.Tensor:
    x = hashes.to(torch.int64)
    c1, c2 = _get_owner_mix_tensors(hashes.device)
    x = x ^ (x >> 33)
    x = x * c1
    x = x ^ (x >> 33)
    x = x * c2
    x = x ^ (x >> 33)
    return x


def _owners_from_hashes(hashes: torch.Tensor, world_size: int) -> torch.Tensor:
    mixed = _owner_mix_hashes(hashes)
    return torch.remainder(mixed, world_size)


def _owner_sorted_spans(owners_sorted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = owners_sorted.shape[0]
    if n == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=owners_sorted.device)
        return empty, empty, empty

    change = torch.ones(n, dtype=torch.bool, device=owners_sorted.device)
    if n > 1:
        change[1:] = owners_sorted[1:] != owners_sorted[:-1]

    starts = torch.nonzero(change, as_tuple=False).reshape(-1)
    owner_ids = owners_sorted[starts]
    ends = torch.empty_like(starts)
    if starts.numel() > 1:
        ends[:-1] = starts[1:]
    ends[-1] = n
    return owner_ids, starts, ends


def _get_free_cuda_bytes(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    except Exception:
        return None


def _infer_expand_chunk_size(
    graph: "CayleyGraph",
    device: torch.device,
    requested_expand_chunk_size: int,
    predictor_batch_size: int,
    local_beam_size: int,
    state_dtype: torch.dtype,
) -> int:
    if requested_expand_chunk_size > 0:
        return requested_expand_chunk_size

    free_bytes = _get_free_cuda_bytes(device)
    if free_bytes is None:
        if predictor_batch_size > 0:
            return predictor_batch_size
        return max(1024, local_beam_size)

    state_width = max(1, int(graph.encoded_state_size))
    state_bytes = state_width * torch.tensor([], dtype=state_dtype).element_size()

    approx_bytes_per_state = (
        state_bytes
        + 8
        + 4
        + 8
        + 8
        + 1
        + state_bytes
    )
    approx_bytes_per_state = max(64, int(approx_bytes_per_state * 1.5))

    safe_budget = max(64 << 20, int(free_bytes * 0.15))
    inferred = safe_budget // max(1, approx_bytes_per_state)
    inferred = max(1024, inferred)

    if predictor_batch_size > 0:
        inferred = max(predictor_batch_size, inferred)

    if local_beam_size > 0:
        inferred = min(inferred, max(local_beam_size, 1024))

    return int(inferred)


def _sync_expand_chunk_size_step(
    local_chunk_size: int,
    local_had_oom: bool,
    base_chunk_size: int,
    device: torch.device,
) -> tuple[int, bool]:
    size_t = torch.tensor([int(local_chunk_size)], dtype=torch.int64, device=device)
    oom_t = torch.tensor([int(local_had_oom)], dtype=torch.int64, device=device)

    dist.all_reduce(size_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(oom_t, op=dist.ReduceOp.MAX)

    synced_size = int(size_t.item())
    any_oom = bool(int(oom_t.item()))

    if base_chunk_size > 0:
        synced_size = min(int(base_chunk_size), synced_size)

    synced_size = max(1, synced_size)
    return synced_size, any_oom


def _maybe_grow_chunk_size(
    current_chunk_size: int,
    base_chunk_size: int,
    success_streak: int,
) -> int:
    if current_chunk_size >= base_chunk_size:
        return current_chunk_size
    if success_streak < 3:
        return current_chunk_size
    grown = int(math.ceil(current_chunk_size * 1.25))
    return min(base_chunk_size, max(current_chunk_size + 1, grown))


def _append_owner_part(
    owner_states_parts: list[list[torch.Tensor]],
    owner_scores_parts: list[list[torch.Tensor]],
    owner_rank: int,
    states: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    if states.numel() == 0:
        return
    owner_states_parts[owner_rank].append(states)
    owner_scores_parts[owner_rank].append(scores)


def _finalize_owner_parts(
    owner_states_parts: list[list[torch.Tensor]],
    owner_scores_parts: list[list[torch.Tensor]],
    owner_budgets: list[int],
    width: int,
    device: torch.device,
    state_dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    world_size = len(owner_budgets)
    final_states = [_empty_states(device, width, state_dtype) for _ in range(world_size)]
    final_scores = [_empty_scores(device) for _ in range(world_size)]

    for owner_rank in range(world_size):
        budget = owner_budgets[owner_rank]
        if budget <= 0:
            continue

        state_parts = owner_states_parts[owner_rank]
        score_parts = owner_scores_parts[owner_rank]
        if not state_parts:
            continue

        cur_states = _empty_states(device, width, state_dtype)
        cur_scores = _empty_scores(device)

        for part_states, part_scores in zip(state_parts, score_parts):
            if part_states.numel() == 0:
                continue

            if cur_states.numel() == 0:
                if part_states.shape[0] <= budget:
                    cur_states = part_states
                    cur_scores = part_scores
                else:
                    cur_states, cur_scores = _topk_by_score(part_states, part_scores, budget)
                continue

            merged_n = cur_states.shape[0] + part_states.shape[0]
            if merged_n <= budget:
                cur_states = torch.cat([cur_states, part_states], dim=0)
                cur_scores = torch.cat([cur_scores, part_scores], dim=0)
                continue

            merged_states = torch.cat([cur_states, part_states], dim=0)
            merged_scores = torch.cat([cur_scores, part_scores], dim=0)
            cur_states, cur_scores = _topk_by_score(merged_states, merged_scores, budget)

        final_states[owner_rank] = cur_states
        final_scores[owner_rank] = cur_scores

    return final_states, final_scores


def _prepare_send_buffers(
    send_states_parts: list[torch.Tensor],
    send_scores_parts: list[torch.Tensor],
    width: int,
    device: torch.device,
    state_dtype: torch.dtype,
    workspace: _SearchWorkspace,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    send_counts = [int(part.shape[0]) for part in send_states_parts]
    total_send = sum(send_counts)

    if total_send == 0:
        return _empty_states(device, width, state_dtype), _empty_scores(device), send_counts

    send_states = workspace.get_send_states(total_send)
    send_scores = workspace.get_send_scores(total_send)

    offset = 0
    for states_part, scores_part in zip(send_states_parts, send_scores_parts):
        n = states_part.shape[0]
        if n == 0:
            continue
        send_states[offset : offset + n].copy_(states_part)
        send_scores[offset : offset + n].copy_(scores_part)
        offset += n

    return send_states[:total_send], send_scores[:total_send], send_counts


def _iter_neighbors_chunks(
    graph: "CayleyGraph",
    beam_chunk: torch.Tensor,
) -> Iterable[torch.Tensor]:
    chunked_fn = getattr(graph, "get_neighbors_chunked", None)
    if callable(chunked_fn):
        for item in chunked_fn(beam_chunk):
            if isinstance(item, tuple) and len(item) == 3:
                _, _, neighbors_chunk = item
            else:
                neighbors_chunk = item
            neighbors_chunk = _normalize_states(neighbors_chunk, graph.encoded_state_size)
            if neighbors_chunk.numel() > 0:
                yield neighbors_chunk
        return

    neighbors = graph.get_neighbors(beam_chunk)
    neighbors = _normalize_states(neighbors, graph.encoded_state_size)
    if neighbors.numel() > 0:
        yield neighbors


def _process_neighbors_chunk(
    graph: "CayleyGraph",
    *,
    neighbors: torch.Tensor,
    destination_encoded: torch.Tensor,
    world_size: int,
    history_hashes: list[torch.Tensor],
    predictor: Optional[Predictor],
    owner_budgets: list[int],
    predictor_batch_size: int,
    device: torch.device,
    workspace: _SearchWorkspace,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], bool]:
    result_parts: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    found_local = False

    if neighbors.numel() == 0:
        return result_parts, False

    neighbors, hashes = graph.get_unique_states(neighbors)
    if neighbors.numel() == 0:
        return result_parts, False

    if torch.any(torch.all(neighbors == destination_encoded, dim=1)).item():
        found_local = True

    history_mask = _filter_hashes_against_history(hashes, history_hashes)
    if not torch.any(history_mask).item():
        return result_parts, found_local

    neighbors = neighbors[history_mask]
    hashes = hashes[history_mask]
    if neighbors.numel() == 0:
        return result_parts, found_local

    owners = _owners_from_hashes(hashes, world_size)
    owner_order = torch.argsort(owners)

    owners = owners[owner_order]
    neighbors = neighbors[owner_order]

    owner_ids, starts, ends = _owner_sorted_spans(owners)

    for span_idx in range(owner_ids.shape[0]):
        owner_rank = int(owner_ids[span_idx].item())
        owner_budget = owner_budgets[owner_rank]
        if owner_budget <= 0:
            continue

        start = int(starts[span_idx].item())
        end = int(ends[span_idx].item())
        owner_chunk_states = neighbors[start:end]
        if owner_chunk_states.numel() == 0:
            continue

        owner_chunk_scores = _score_states_oom_safe(
            graph,
            owner_chunk_states,
            predictor,
            device,
            predictor_batch_size=predictor_batch_size,
            workspace=workspace,
        )
        owner_chunk_states, owner_chunk_scores = _topk_by_score(
            owner_chunk_states,
            owner_chunk_scores,
            owner_budget,
        )
        result_parts.append((owner_rank, owner_chunk_states, owner_chunk_scores))

    return result_parts, found_local


def _process_beam_chunk_once(
    graph: "CayleyGraph",
    *,
    beam_chunk: torch.Tensor,
    destination_encoded: torch.Tensor,
    world_size: int,
    history_hashes: list[torch.Tensor],
    predictor: Optional[Predictor],
    owner_budgets: list[int],
    predictor_batch_size: int,
    device: torch.device,
    workspace: _SearchWorkspace,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], bool]:
    result_parts: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    found_local = False

    for neighbors_chunk in _iter_neighbors_chunks(graph, beam_chunk):
        chunk_parts, chunk_found = _process_neighbors_chunk(
            graph,
            neighbors=neighbors_chunk,
            destination_encoded=destination_encoded,
            world_size=world_size,
            history_hashes=history_hashes,
            predictor=predictor,
            owner_budgets=owner_budgets,
            predictor_batch_size=predictor_batch_size,
            device=device,
            workspace=workspace,
        )
        if chunk_parts:
            result_parts.extend(chunk_parts)
        found_local = found_local or chunk_found

    return result_parts, found_local


def _owner_partitioned_streaming_candidates(
    graph: "CayleyGraph",
    *,
    local_beam_states: torch.Tensor,
    destination_encoded: torch.Tensor,
    beam_width: int,
    world_size: int,
    history_hashes: list[torch.Tensor],
    predictor: Optional[Predictor],
    oversubscription_factor: float,
    expand_chunk_size: int,
    predictor_batch_size: int,
    device: torch.device,
    workspace: _SearchWorkspace,
) -> tuple[list[torch.Tensor], list[torch.Tensor], bool, int, bool]:
    width = graph.encoded_state_size
    state_dtype = local_beam_states.dtype if local_beam_states.numel() > 0 else workspace.state_dtype
    found_local = False

    if local_beam_states.numel() == 0:
        inferred_chunk = _infer_expand_chunk_size(
            graph,
            device,
            expand_chunk_size,
            predictor_batch_size,
            0,
            state_dtype,
        )
        empty_states = [_empty_states(device, width, state_dtype) for _ in range(world_size)]
        empty_scores = [_empty_scores(device) for _ in range(world_size)]
        return empty_states, empty_scores, False, inferred_chunk, False

    base_chunk_size = _infer_expand_chunk_size(
        graph,
        device,
        expand_chunk_size,
        predictor_batch_size,
        local_beam_states.shape[0],
        state_dtype,
    )
    current_chunk_size = base_chunk_size

    owner_budgets = [
        _compute_owner_pre_k(
            beam_width=beam_width,
            world_size=world_size,
            owner_rank=owner_rank,
            oversubscription_factor=oversubscription_factor,
        )
        for owner_rank in range(world_size)
    ]

    owner_states_parts: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    owner_scores_parts: list[list[torch.Tensor]] = [[] for _ in range(world_size)]

    beam_offset = 0
    success_streak = 0
    had_local_oom = False

    while beam_offset < local_beam_states.shape[0]:
        local_parts: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        local_found = False
        upper = min(local_beam_states.shape[0], beam_offset + current_chunk_size)
        beam_chunk = local_beam_states[beam_offset:upper]

        try:
            local_parts, local_found = _process_beam_chunk_once(
                graph,
                beam_chunk=beam_chunk,
                destination_encoded=destination_encoded,
                world_size=world_size,
                history_hashes=history_hashes,
                predictor=predictor,
                owner_budgets=owner_budgets,
                predictor_batch_size=predictor_batch_size,
                device=device,
                workspace=workspace,
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            had_local_oom = True
            _cleanup_after_oom(device)
            current_chunk_size = max(1, current_chunk_size // 2)
            success_streak = 0
            continue

        for owner_rank, owner_chunk_states, owner_chunk_scores in local_parts:
            _append_owner_part(
                owner_states_parts,
                owner_scores_parts,
                owner_rank,
                owner_chunk_states,
                owner_chunk_scores,
            )

        found_local = found_local or local_found
        beam_offset = upper
        success_streak += 1
        current_chunk_size = _maybe_grow_chunk_size(current_chunk_size, base_chunk_size, success_streak)

    final_states, final_scores = _finalize_owner_parts(
        owner_states_parts,
        owner_scores_parts,
        owner_budgets,
        width,
        device,
        state_dtype,
    )
    return final_states, final_scores, found_local, current_chunk_size, had_local_oom


def search_multigpu(
    graph: "CayleyGraph",
    *,
    start_state: AnyStateType,
    destination_state: Optional[AnyStateType] = None,
    beam_width: int = 1000,
    max_steps: int = 1000,
    history_depth: int = 0,
    predictor: Optional[Predictor] = None,
    oversubscription_factor: float = 2.0,
    expand_chunk_size: int = 4096,
    predictor_batch_size: int = 4096,
    verbose: int = 0,
) -> BeamSearchResult:
    """Run beam search with a torchrun distributed path when WORLD_SIZE > 1.

    Single-process execution delegates to `BeamSearchAlgorithm.search_advanced`.
    Distributed execution uses only owner-partitioned CUDA/NCCL routing.
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

    return search_multigpu_owner_partitioned(
        graph,
        start_state=start_state,
        destination_state=destination_state,
        beam_width=beam_width,
        max_steps=max_steps,
        history_depth=history_depth,
        predictor=predictor,
        oversubscription_factor=oversubscription_factor,
        expand_chunk_size=expand_chunk_size,
        predictor_batch_size=predictor_batch_size,
        verbose=verbose,
    )


def search_multigpu_owner_partitioned(
    graph: "CayleyGraph",
    *,
    start_state: AnyStateType,
    destination_state: Optional[AnyStateType] = None,
    beam_width: int = 1000,
    max_steps: int = 1000,
    history_depth: int = 0,
    predictor: Optional[Predictor] = None,
    oversubscription_factor: float = 2.0,
    expand_chunk_size: int = 4096,
    predictor_batch_size: int = 4096,
    verbose: int = 0,
) -> BeamSearchResult:
    """Scalable CUDA/NCCL torchrun beam search with owner-based state partitioning."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if oversubscription_factor < 1.0:
        raise ValueError("oversubscription_factor must be >= 1.0")
    if expand_chunk_size < 0:
        raise ValueError("expand_chunk_size must be >= 0")
    if predictor_batch_size < 0:
        raise ValueError("predictor_batch_size must be >= 0")

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

    try:
        width = graph.encoded_state_size
        start_encoded = _encode_states_to_device(graph, start_state, device)
        state_dtype = start_encoded.dtype

        workspace = _SearchWorkspace(
            device=device,
            state_dtype=state_dtype,
            width=width,
        )

        destination_effective = destination_state if destination_state is not None else graph.central_state
        destination_encoded = _encode_states_to_device(graph, destination_effective, device)

        start_found_local = bool(torch.any(torch.all(start_encoded == destination_encoded, dim=1)).item())
        start_found = torch.tensor([int(start_found_local)], dtype=torch.int64, device=device)
        dist.all_reduce(start_found, op=dist.ReduceOp.MAX)
        if int(start_found.item()) > 0:
            return BeamSearchResult(True, 0, [], {}, graph.definition)

        start_states, start_hashes = graph.get_unique_states(start_encoded)
        start_owners = _owners_from_hashes(start_hashes, world_size)
        start_mask = start_owners == rank

        local_beam_states = start_states[start_mask]
        local_beam_hashes = start_hashes[start_mask]

        local_target = _compute_per_rank_beam(beam_width, world_size, rank)
        if local_beam_states.shape[0] > local_target > 0:
            seed_scores = _score_states_oom_safe(
                graph,
                local_beam_states,
                predictor,
                device,
                predictor_batch_size=predictor_batch_size,
                workspace=workspace,
            )
            local_beam_states, _ = _topk_by_score(local_beam_states, seed_scores, local_target)
            local_beam_hashes = graph.hasher.make_hashes(local_beam_states)
        elif local_beam_states.shape[0] == 0:
            local_beam_hashes = _empty_hashes(device)

        history_hashes = (
            [_sorted_unique_hashes(local_beam_hashes.detach())]
            if history_depth > 0 and local_beam_hashes.numel() > 0
            else []
        )
        debug_scores: dict[int, float] = {}
        adaptive_expand_chunk_size = expand_chunk_size

        for step in range(1, max_steps + 1):
            (
                send_states_parts,
                send_scores_parts,
                found_local,
                used_chunk_size,
                had_local_oom,
            ) = _owner_partitioned_streaming_candidates(
                graph,
                local_beam_states=local_beam_states,
                destination_encoded=destination_encoded,
                beam_width=beam_width,
                world_size=world_size,
                history_hashes=history_hashes,
                predictor=predictor,
                oversubscription_factor=oversubscription_factor,
                expand_chunk_size=adaptive_expand_chunk_size,
                predictor_batch_size=predictor_batch_size,
                device=device,
                workspace=workspace,
            )

            adaptive_expand_chunk_size, any_oom_this_step = _sync_expand_chunk_size_step(
                used_chunk_size,
                had_local_oom,
                adaptive_expand_chunk_size,
                device,
            )

            if verbose >= 2 and rank == 0 and any_oom_this_step:
                print(f"OOM shrink: new_expand_chunk_size={adaptive_expand_chunk_size}.")

            send_states, send_scores, send_counts = _prepare_send_buffers(
                send_states_parts,
                send_scores_parts,
                width,
                device,
                state_dtype,
                workspace,
            )

            send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
            gathered_counts = torch.empty((world_size * world_size,), dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(gathered_counts, send_counts_t)

            count_matrix = gathered_counts.view(world_size, world_size)
            recv_counts_t = count_matrix[:, rank].contiguous()
            recv_counts = [int(x) for x in recv_counts_t.tolist()]
            total_recv = int(recv_counts_t.sum().item())

            recv_states = workspace.get_recv_states(total_recv)
            recv_scores = workspace.get_recv_scores(total_recv)

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

            unique_states, unique_hashes, unique_scores = _deduplicate_keep_best_score(
                graph,
                recv_states[:total_recv],
                recv_scores[:total_recv],
            )

            if history_depth > 0:
                unique_states, unique_hashes, unique_scores = _filter_history(
                    unique_states,
                    unique_hashes,
                    unique_scores,
                    history_hashes,
                )

            local_beam_states, next_scores = _topk_by_score(unique_states, unique_scores, local_target)

            if local_beam_states.numel() == 0:
                local_beam_hashes = _empty_hashes(device)
            else:
                local_beam_hashes = graph.hasher.make_hashes(local_beam_states)

            history_hashes = _update_history(history_hashes, local_beam_hashes, history_depth)

            local_best = float(_topk_scores_only(next_scores, 1).min().item()) if next_scores.numel() > 0 else float("inf")
            stop_code, best_score = _reduce_step_status(
                found_local,
                local_beam_states.shape[0] > 0,
                local_best,
                device,
            )

            if math.isfinite(best_score):
                debug_scores[step] = best_score

            if verbose >= 2:
                local_count_t = torch.tensor([int(local_beam_states.shape[0])], dtype=torch.int64, device=device)
                global_count_t = local_count_t.clone()
                dist.all_reduce(global_count_t, op=dist.ReduceOp.SUM)

                send_count_total_t = torch.tensor([sum(send_counts)], dtype=torch.int64, device=device)
                global_send_total_t = send_count_total_t.clone()
                dist.all_reduce(global_send_total_t, op=dist.ReduceOp.SUM)

                owner_load_t = torch.tensor([int(local_beam_states.shape[0])], dtype=torch.int64, device=device)
                owner_load_max_t = owner_load_t.clone()
                owner_load_min_t = owner_load_t.clone()
                dist.all_reduce(owner_load_max_t, op=dist.ReduceOp.MAX)
                dist.all_reduce(owner_load_min_t, op=dist.ReduceOp.MIN)

                if rank == 0:
                    min_load = int(owner_load_min_t.item())
                    max_load = int(owner_load_max_t.item())
                    imbalance = (max_load / max(1, min_load)) if min_load > 0 else float("inf")
                    print(
                        f"Step {step}: "
                        f"beam={int(global_count_t.item())}, "
                        f"best_score={best_score:.6f}, "
                        f"sent_preprune={int(global_send_total_t.item())}, "
                        f"owner_load_min={min_load}, "
                        f"owner_load_max={max_load}, "
                        f"owner_imbalance={imbalance:.3f}, "
                        f"expand_chunk_size={adaptive_expand_chunk_size}."
                    )

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

    finally:
        _cleanup_distributed_context()


__all__ = [
    "search_multigpu",
    "search_multigpu_owner_partitioned",
]
