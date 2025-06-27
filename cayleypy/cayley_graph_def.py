from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Union

import numpy as np
import torch

from .permutation_utils import inverse_permutation


@dataclass(frozen=True)
class CayleyGraphDef:
    """Mathematical definition of a CayleyGraph."""

    generators: list[list[int]]
    generator_names: list[str]
    central_state: list[int]

    @staticmethod
    def create(
        generators: Union[list[list[int]], torch.Tensor, np.ndarray],
        generator_names: Optional[list[str]] = None,
        central_state: Union[list[int], torch.Tensor, np.ndarray, str, None] = None,
    ):
        """Creates Cayley Graph definition.

        :param generators: List of generating permutations of size n.
        :param generator_names: Names of the generators (optional).
        :param central_state: List of n numbers between 0 and n-1, the central state.
                 If None, defaults to the identity permutation of size n.
        """
        # Prepare generators.
        if isinstance(generators, list):
            generators_list = generators
        elif isinstance(generators, torch.Tensor):
            generators_list = [[q.item() for q in generators[i, :]] for i in range(generators.shape[0])]
        elif isinstance(generators, np.ndarray):
            generators_list = [list(generators[i, :]) for i in range(generators.shape[0])]
        else:
            raise ValueError('Unsupported format for "generators" ' + str(type(generators)))

        # Validate generators.
        n = len(generators_list[0])
        id_perm = list(range(n))
        for perm in generators_list:
            assert sorted(perm) == id_perm, f"{perm} is not a permutation of length {n}."

        # Prepare generator names.
        if generator_names is None:
            generator_names = [",".join(str(i) for i in g) for g in generators_list]

        # Prepare and validate central state.
        if central_state is None:
            central_state = list(range(n))  # Identity permutation.
        else:
            central_state = [int(x) for x in central_state]
        assert len(central_state) == n
        assert min(central_state) >= 0
        assert max(central_state) < n

        return CayleyGraphDef(generators_list, generator_names, central_state)

    @cached_property
    def n_generators(self) -> int:
        return len(self.generators)

    @cached_property
    def state_size(self) -> int:
        return len(self.central_state)

    @cached_property
    def generators_inverse_closed(self) -> bool:
        """Whether for each generator its inverse is also a generator."""
        generators_set = set(tuple(perm) for perm in self.generators)
        for perm in self.generators:
            if tuple(inverse_permutation(perm)) not in generators_set:
                return False
        return True

    def with_central_state(self, central_state) -> "CayleyGraphDef":
        return CayleyGraphDef.create(self.generators, generator_names=self.generator_names, central_state=central_state)
