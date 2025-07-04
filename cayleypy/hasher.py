from typing import Callable, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cayleypy import CayleyGraph


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
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.vec_hasher = torch.randint(
            -(2**63), 2**63 - 1, size=(self.state_size,), device=graph.device, dtype=torch.int64
        )
        self.seed = self.vec_hasher[0]

        if graph.string_encoder is not None:
            # Dot product is not safe for bit-encoded states, it has high probability of collisions.
            self.make_hashes = self._hash_combine
            return

        self.make_hashes = self._hash_dot_product

    def _hash_dot_product(self, states: torch.Tensor) -> torch.Tensor:
        return torch.sum(states * self.vec_hasher, dim=1)

    def _hash_combine(self, states: torch.Tensor) -> torch.Tensor:
        """Hash function inspired by boost::hash_combine."""
        result = states[:, 0].clone()
        for i in range(1, self.state_size):
            result ^= states[:, i] + self.seed + (result << 6) + (result >> 2)
        return result
