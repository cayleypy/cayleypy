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
        self.seed = random_seed or random.randint(-MAX_INT, MAX_INT)

        # Dot product is not safe for bit-encoded states, it has high probability of collisions.
        if graph.string_encoder is not None:
            self.make_hashes = self._hash_splitmix64
            return

        torch.manual_seed(self.seed)
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

    def _make_hashes_cpu_and_modern_gpu(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[0] <= self.chunk_size:
            return (states @ self.vec_hasher).reshape(-1)
        else:
            parts = int(math.ceil(states.shape[0] / self.chunk_size))
            return torch.vstack([z @ self.vec_hasher for z in torch.tensor_split(states, parts)]).reshape(-1)

    def _make_hashes_older_gpu(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[0] <= self.chunk_size:
            return torch.sum(states * self.vec_hasher, dim=1)
        else:
            parts = int(math.ceil(states.shape[0] / self.chunk_size))
            return torch.hstack([torch.sum(z * self.vec_hasher, dim=1) for z in torch.tensor_split(states, parts)])

    def _hash_splitmix64(self, x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        h = torch.full((n,), self.seed, dtype=torch.int64, device=x.device)
        for i in range(m):
            h ^= _splitmix64(x[:, i])
            h = h * 0x85EBCA6B
        return h
