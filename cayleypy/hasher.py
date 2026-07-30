import math
import random
from typing import Callable, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cayleypy import CayleyGraph

MAX_INT = 2**62


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x


class StateHasher:
    """Helper class to hash states."""

    def __init__(self, graph: "CayleyGraph", random_seed: Optional[int], chunk_size=2**18):
        self.state_size = graph.encoded_state_size
        self.chunk_size = chunk_size

        # If states are already encoded by a single int64, use identity function as hash function.
        self.make_hashes: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.reshape(-1)
        self.is_identity = True
        if self.state_size == 1:
            return

        self.is_identity = False
        self.seed = random_seed if random_seed is not None else random.randint(-MAX_INT, MAX_INT)

        # Dot product is not safe for bit-encoded states, it has high probability of collisions.
        if graph.string_encoder is not None:
            self.make_hashes = self._hash_splitmix64
            return

        torch.manual_seed(self.seed)

        # CPU-only fast path (permutation groups only): dual int32 matmul combined to
        # 2^64 via a zero-copy view. ~1.1-1.2x faster than the int64 path on CPU (cheaper
        # int8/int64->int32 cast, BLAS-optimized int32 matmul, zero-copy combine). GPU
        # keeps the int64 path below because dual int32 doubles kernel launches there.
        # The two int32 hashes are independent, so a collision requires both to collide
        # (~2^-64 per pair).
        # Gated on permutation groups: permutation state values are small indices (< n),
        # so the int64->int32 cast is lossless. Matrix groups (modulo==0) can hold arbitrary
        # int64 values whose high bits would be silently truncated by the int32 cast, so
        # they must keep the int64 path to preserve dedup correctness.
        if graph.device.type == "cpu" and graph.definition.is_permutation_group():
            self.vec_hasher_i32 = torch.randint(
                -(2**30), 2**30, size=(self.state_size, 2), device=graph.device, dtype=torch.int32
            )
            self.make_hashes = self._make_hashes_dual_int32
            return

        # GPU path (and CPU matrix groups): int64 matmul (BLAS on modern GPUs,
        # sum-reduction fallback on older GPUs).
        self.vec_hasher = torch.randint(
            -MAX_INT, MAX_INT, size=(self.state_size, 1), device=graph.device, dtype=torch.int64
        )

        try:
            trial_states = torch.zeros((2, self.state_size), device=graph.device, dtype=torch.int64)
            _ = self._make_hashes_cpu_and_modern_gpu(trial_states)
            self.make_hashes = self._make_hashes_cpu_and_modern_gpu
        except RuntimeError:
            self.vec_hasher = self.vec_hasher.reshape((self.state_size,))
            self.make_hashes = self._make_hashes_older_gpu

    def _make_hashes_dual_int32(self, states: torch.Tensor) -> torch.Tensor:
        # One int32 matmul against a (state_size, 2) matrix yields two independent
        # int32 hashes in a single (n, 2) tensor, which is byte-reinterpretable as
        # int64 via a zero-copy ``.view`` (2 x int32 == 1 x int64). This reads the
        # casted states once instead of two separate matmuls. On little-endian
        # (x86/ARM) the second hash lands in the high 32 bits; this is a valid
        # deterministic hash (same input -> same output on one architecture).
        if states.shape[0] <= self.chunk_size:
            s32 = states.to(torch.int32)
            return (s32 @ self.vec_hasher_i32).view(torch.int64).reshape(-1)
        parts = int(math.ceil(states.shape[0] / self.chunk_size))
        result = []
        for z in torch.tensor_split(states, parts):
            s32 = z.to(torch.int32)
            result.append((s32 @ self.vec_hasher_i32).view(torch.int64).reshape(-1))
        return torch.hstack(result)

    def _make_hashes_cpu_and_modern_gpu(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[0] <= self.chunk_size:
            return (states.to(torch.int64) @ self.vec_hasher).reshape(-1)
        else:
            parts = int(math.ceil(states.shape[0] / self.chunk_size))
            return torch.vstack(
                [z.to(torch.int64) @ self.vec_hasher for z in torch.tensor_split(states, parts)]
            ).reshape(-1)

    def _make_hashes_older_gpu(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[0] <= self.chunk_size:
            return torch.sum(states.to(torch.int64) * self.vec_hasher, dim=1)
        else:
            parts = int(math.ceil(states.shape[0] / self.chunk_size))
            return torch.hstack(
                [torch.sum(z.to(torch.int64) * self.vec_hasher, dim=1) for z in torch.tensor_split(states, parts)]
            )

    def _hash_splitmix64(self, x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        h = torch.full((n,), self.seed, dtype=torch.int64, device=x.device)
        for i in range(m):
            h ^= _splitmix64(x[:, i].to(torch.int64))
            h = h * 0x85EBCA6B
        return h
