from __future__ import annotations

import math
import os
import threading
from typing import Optional, Sequence, Union

import torch

from .cayley_graph import CayleyGraph
from .cayley_graph_def import AnyStateType, CayleyGraphDef, GeneratorType
from .device_config import DeviceConfig
from .hasher import StateHasher

# ВАЖНО:
# Файл ожидает, что beam_search_multigpu.py уже существует в .algo
# и экспортирует search_multigpu.
from .algo.beam_search_multigpu import search_multigpu


class _ReusableStateBuffer:
    """Grow-only reusable 2D tensor buffer."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._buf: Optional[torch.Tensor] = None
        self._rows: int = 0
        self._cols: int = 0
        self._lock = threading.Lock()

    def reserve(self, rows: int, cols: int) -> torch.Tensor:
        rows = max(0, int(rows))
        cols = max(0, int(cols))
        with self._lock:
            need_new = (
                self._buf is None
                or self._rows < rows
                or self._cols < cols
                or self._buf.device != self.device
                or self._buf.dtype != self.dtype
            )
            if need_new:
                grow_rows = max(rows, int(math.ceil(max(1, rows) * 1.25)))
                grow_cols = max(cols, self._cols, 1)
                self._buf = torch.empty(
                    (grow_rows, grow_cols),
                    dtype=self.dtype,
                    device=self.device,
                )
                self._rows = grow_rows
                self._cols = grow_cols
            return self._buf[:rows, :cols]


class CayleyGraph_Multi(CayleyGraph):
    """Drop-in replacement for CayleyGraph for multi-GPU beam search.

    Goals:
    - preserve public API compatibility;
    - reduce CUDA allocator churn;
    - reuse neighbor buffers;
    - optionally dispatch beam_search() into beam_search_multigpu.search_multigpu.
    """

    def __init__(
        self,
        definition: CayleyGraphDef,
        *,
        device: str = "auto",
        random_seed: Optional[int] = None,
        bit_encoding_width: Union[Optional[int], str] = "auto",
        verbose: int = 0,
        batch_size: int = 2**20,
        hash_chunk_size: int = 2**25,
        memory_limit_gb: float = 16,
        num_gpus: Optional[int] = None,
        specific_devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
        device_config: Optional[DeviceConfig] = None,
        _hasher: Optional[StateHasher] = None,
        neighbors_generator_chunk_size: int = 4,
        prefer_multigpu_beam_search: bool = True,
        free_memory_on_large_resize: bool = False,
        **unused_kwargs,
    ):
        super().__init__(
            definition,
            device=device,
            random_seed=random_seed,
            bit_encoding_width=bit_encoding_width,
            verbose=verbose,
            batch_size=batch_size,
            hash_chunk_size=hash_chunk_size,
            memory_limit_gb=memory_limit_gb,
            num_gpus=num_gpus,
            specific_devices=specific_devices,
            device_config=device_config,
            _hasher=_hasher,
            **unused_kwargs,
        )

        # Сколько генераторов materialize за один внутренний подшаг.
        # Значение 1..4 обычно снижает пиковую память allocator-а.
        self.neighbors_generator_chunk_size = max(1, int(neighbors_generator_chunk_size))

        # Переключатель: beam_search() -> search_multigpu(...)
        self.prefer_multigpu_beam_search = bool(prefer_multigpu_beam_search)

        # Опционально чистить cache при сильном росте буфера.
        self.free_memory_on_large_resize = bool(free_memory_on_large_resize)

        # Буфер соседей.
        self._neighbors_buffer = _ReusableStateBuffer(
            device=self.device,
            dtype=torch.int64,
        )

        # Временный буфер для apply_generator_batched по чанкам генераторов.
        self._tmp_generator_buffer = _ReusableStateBuffer(
            device=self.device,
            dtype=torch.int64,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _maybe_free_after_resize(self) -> None:
        if not self.free_memory_on_large_resize:
            return
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _state_width(self, states: torch.Tensor) -> int:
        if states.dim() == 1:
            return int(states.shape[0])
        return int(states.shape[1])

    def _get_neighbors_identity_hasher_fastpath(self, states: torch.Tensor) -> torch.Tensor:
        """Fast path for identity hash graphs with width==1 semantics."""
        states_num = states.shape[0]
        n_generators = self.definition.n_generators
        width = states.shape[1]

        out = self._neighbors_buffer.reserve(states_num * n_generators, width)

        for g in range(n_generators):
            dst = out[g * states_num : (g + 1) * states_num]
            self.apply_generator_batched(g, states, dst)

        return out

    # -------------------------------------------------------------------------
    # API overrides
    # -------------------------------------------------------------------------

    def get_unique_states(
        self,
        states: torch.Tensor,
        hashes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states` and returns one representative per hash.
    
        Properties:
        - single sort by hash
        - no stable sort
        - no extra reorder beyond survivor gather
        """
        if states.numel() == 0:
            width = self.encoded_state_size if states.dim() != 2 else states.shape[1]
            return (
                torch.empty((0, width), dtype=states.dtype, device=states.device),
                torch.empty((0,), dtype=torch.int64, device=states.device),
            )
    
        if self.hasher.is_identity:
            unique_hashes = torch.unique(states.reshape(-1), sorted=True)
            return unique_hashes.reshape((-1, 1)), unique_hashes
    
        if hashes is None:
            hashes = self.hasher.make_hashes(states)
    
        if hashes.numel() <= 1:
            return states, hashes
    
        order = torch.argsort(hashes)
        hashes_sorted = hashes[order]
    
        keep = torch.ones(hashes_sorted.shape[0], dtype=torch.bool, device=hashes_sorted.device)
        keep[1:] = hashes_sorted[1:] != hashes_sorted[:-1]
    
        unique_idx = order[keep]
        return states[unique_idx], hashes[unique_idx]

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates all neighbors of states in internal representation.

        Differences from CayleyGraph.get_neighbors:
        - reusable output buffer;
        - optional generator-chunked filling;
        - no repeated fresh torch.zeros allocation per call.
        """
        if states.dim() != 2:
            states = states.reshape((-1, self.encoded_state_size))

        states_num = int(states.shape[0])
        width = int(states.shape[1])
        n_generators = int(self.definition.n_generators)

        if states_num == 0:
            return torch.empty((0, width), dtype=torch.int64, device=states.device)

        total_rows = states_num * n_generators
        neighbors = self._neighbors_buffer.reserve(total_rows, width)

        # Вариант без chunking по генераторам.
        if self.neighbors_generator_chunk_size >= n_generators:
            for i in range(n_generators):
                dst = neighbors[i * states_num : (i + 1) * states_num]
                self.apply_generator_batched(i, states, dst)
            return neighbors

        # Chunked fill. Пиковая память ниже в сценариях, где apply_generator_batched
        # внутри порождает временные буферы, зависящие от числа генераторов/запусков.
        g0 = 0
        while g0 < n_generators:
            g1 = min(n_generators, g0 + self.neighbors_generator_chunk_size)
            for i in range(g0, g1):
                dst = neighbors[i * states_num : (i + 1) * states_num]
                self.apply_generator_batched(i, states, dst)
            g0 = g1

        return neighbors

    def get_neighbors_chunked(
        self,
        states: torch.Tensor,
        *,
        generators_per_chunk: Optional[int] = None,
    ):
        """Yield neighbors in generator chunks.

        This method is additive.
        This method does not affect old code.
        beam_search_multigpu.py can optionally use this method in future.

        Yields:
            tuple[int, int, torch.Tensor]
            (g_start, g_end, neighbors_chunk)
        """
        if states.dim() != 2:
            states = states.reshape((-1, self.encoded_state_size))

        states_num = int(states.shape[0])
        width = int(states.shape[1])
        n_generators = int(self.definition.n_generators)

        if states_num == 0:
            empty = torch.empty((0, width), dtype=torch.int64, device=states.device)
            yield 0, 0, empty
            return

        chunk = max(1, int(generators_per_chunk or self.neighbors_generator_chunk_size))
        chunk = min(chunk, n_generators)

        g0 = 0
        while g0 < n_generators:
            g1 = min(n_generators, g0 + chunk)
            rows = (g1 - g0) * states_num
            tmp = self._tmp_generator_buffer.reserve(rows, width)

            for local_idx, gen_idx in enumerate(range(g0, g1)):
                dst = tmp[local_idx * states_num : (local_idx + 1) * states_num]
                self.apply_generator_batched(gen_idx, states, dst)

            yield g0, g1, tmp[:rows]
            g0 = g1

    def beam_search(self, **kwargs):
        """Beam search entry point.

        Behavior:
        - if prefer_multigpu_beam_search=True and torchrun distributed env exists,
          dispatch into beam_search_multigpu.search_multigpu;
        - otherwise preserve original behavior.
        """
        if self.prefer_multigpu_beam_search:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1 and self.device.type == "cuda":
                return search_multigpu(self, **kwargs)

        return super().beam_search(**kwargs)

    def distributed_beam_search(self, **kwargs) -> BeamSearchResult:
        """Explicit distributed beam-search entry point."""
        return search_multigpu(self, **kwargs)

    def free_memory(self):
        """Free graph-owned reusable buffers plus base-class caches."""
        self._neighbors_buffer._buf = None
        self._neighbors_buffer._rows = 0
        self._neighbors_buffer._cols = 0

        self._tmp_generator_buffer._buf = None
        self._tmp_generator_buffer._rows = 0
        self._tmp_generator_buffer._cols = 0

        super().free_memory()

    # -------------------------------------------------------------------------
    # Convenience constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_graph(
        cls,
        graph: CayleyGraph,
        *,
        neighbors_generator_chunk_size: int = 4,
        prefer_multigpu_beam_search: bool = True,
        free_memory_on_large_resize: bool = False,
    ) -> "CayleyGraph_Multi":
        """Create CayleyGraph_Multi from an existing CayleyGraph."""
        new_graph = cls(
            graph.definition,
            device_config=graph.device_config,
            _hasher=graph.hasher,
            bit_encoding_width=graph.bit_encoding_width,
            verbose=graph.verbose,
            batch_size=graph.batch_size,
            hash_chunk_size=getattr(graph.hasher, "chunk_size", 2**25),
            memory_limit_gb=graph.memory_limit_bytes / float(2**30),
            neighbors_generator_chunk_size=neighbors_generator_chunk_size,
            prefer_multigpu_beam_search=prefer_multigpu_beam_search,
            free_memory_on_large_resize=free_memory_on_large_resize,
        )

        # Shared encoder/hasher semantics with original graph.
        new_graph.string_encoder = graph.string_encoder
        new_graph.encoded_state_size = graph.encoded_state_size
        new_graph.central_state = graph.central_state
        new_graph.central_state_hash = graph.central_state_hash
        return new_graph


