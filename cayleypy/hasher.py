import math
from typing import Callable, Optional

import torch


class StateHasher:
    """Helper class to hash states by multiplying them by random vector."""

    def __init__(self, state_size: int, random_seed: Optional[int], device: torch.device, chunk_size=2**18):
        self.state_size = state_size
        self.chunk_size = chunk_size

        # If states are already encoded by a single int64, use identity function as hash function.
        self.make_hashes: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.reshape(-1)
        self.is_identity = True
        if state_size == 1:
            return

        self.is_identity = False
        if random_seed is not None:
            torch.manual_seed(random_seed)
        max_int = int((2**62))
        self.vec_hasher = torch.randint(-max_int, max_int + 1, size=(state_size, 1), device=device, dtype=torch.int64)

        try:
            trial_states = torch.zeros((2, state_size), device=device, dtype=torch.int64)
            _ = self._make_hashes_cpu_and_modern_gpu(trial_states)
            self.make_hashes = self._make_hashes_cpu_and_modern_gpu
        except RuntimeError:
            self.vec_hasher = self.vec_hasher.reshape((state_size,))
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
