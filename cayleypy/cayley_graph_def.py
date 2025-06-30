from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Optional, Union

import numpy as np
import torch

from .permutation_utils import inverse_permutation


class GeneratorType(Enum):
    """Type of generators for Cayley graph."""

    # Generators are permutations of size n applied to vectors of n elements.
    # In this case, the Cayley graph is for group of permutations (S_n).
    PERMUTATION = 1

    # Generators are n*n integer matrices, applied (by multiplication) to n*m matrices.
    # In this case, the Cayley graph is for group of integer square n*n matrices.
    MATRIX = 2


@dataclass(frozen=True)
class MatrixGenerator:
    """Cayley graph generator that is square (n*n) integer matrix.

    This matrix applied (by multiplication) to n*m matrices.
    If `modulo != 0`, multiplication is modulo this number (`2<=modulo<=2^31`).
    If `modulo == 0`, multiplication is signed int64 multiplication with overflow.
    """

    matrix: np.ndarray
    modulo: int

    @staticmethod
    def create(matrix: Union[list, np.ndarray], modulo: int = 0):
        matrix = np.array(matrix, dtype=np.int64)
        if modulo > 0:
            matrix %= modulo
        return MatrixGenerator(matrix, modulo)

    def __post_init__(self):
        # Validation.
        assert self.matrix.shape == (self.n, self.n), "Must be square matrix"
        assert self.matrix.dtype == np.int64
        if self.modulo != 0:
            assert 2 <= self.modulo <= 2**31
            assert self.matrix.min() >= 0
            assert self.matrix.max() < self.modulo

    def is_inverse_to(self, other: "MatrixGenerator") -> bool:
        if self.modulo != other.modulo:
            return False
        eye = np.eye(self.n, dtype=np.int64)
        return np.array_equal(self.apply(other.matrix), eye) and np.array_equal(other.apply(self.matrix), eye)

    @cached_property
    def n(self):
        return self.matrix.shape[0]

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Multiplies (from left) this matrix by a n*m matrix."""
        ans = self.matrix @ state
        if self.modulo > 0:
            ans %= self.modulo
        return ans

    def apply_batch_torch(self, states: torch.Tensor) -> torch.Tensor:
        """Multiplies (from left) this matrix by a batch of n*m torch Tensors."""
        assert len(states.shape) == 3
        assert states.shape[1] == self.n
        mx = torch.tensor(self.matrix, dtype=torch.int64, device=states.device)
        mx = mx.unsqueeze(0).unsqueeze(-1)
        ans = (mx * states.unsqueeze(1)).sum(dim=2)
        if self.modulo > 0:
            ans %= self.modulo
        return ans


@dataclass(frozen=True)
class CayleyGraphDef:
    """Mathematical definition of a CayleyGraph."""

    generators_type: GeneratorType
    generators_permutations: list[list[int]]
    generators_matrices: list[MatrixGenerator]
    generator_names: list[str]
    central_state: list[int]

    @staticmethod
    def create(
        generators: Union[list[list[int]], torch.Tensor, np.ndarray],
        generator_names: Optional[list[str]] = None,
        central_state: Union[list[int], torch.Tensor, np.ndarray, str, None] = None,
    ):
        """Creates Cayley Graph definition (when generators are permutations).

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

        # Prepare central state.
        if central_state is None:
            central_state = list(range(n))  # Identity permutation.
        else:
            central_state = CayleyGraphDef.normalize_central_state(central_state)

        return CayleyGraphDef(GeneratorType.PERMUTATION, generators_list, [], generator_names, central_state)

    @staticmethod
    def for_matrix_group(
        *,
        generators: list[MatrixGenerator],
        generator_names: Optional[list[str]] = None,
        central_state: Union[np.ndarray, list[list[int]], None] = None,
    ):
        """Creates Cayley Graph definition (when generators are matrices).

        :param generators: List of generating n*n matrices.
        :param generator_names: Names of the generators (optional).
        :param central_state: the central state (n*m matrix). Defaults to the n*n identity matrix.
        """
        if generator_names is None:
            generator_names = ["g" + str(i) for i in range(len(generators))]
        if central_state is None:
            # By default, central element is the identity matrix.
            central_state = np.eye(generators[0].n, dtype=np.int64)
        else:
            central_state = np.array(central_state)
            assert len(central_state.shape) == 2, "Central state must be a matrix."
            n = generators[0].n
            assert central_state.shape[0] == n, f"Central state must have shape {n}*m."
        central_state_list = CayleyGraphDef.normalize_central_state(central_state)
        return CayleyGraphDef(GeneratorType.MATRIX, [], generators, generator_names, central_state_list)

    def __post_init__(self):
        # Validation.
        assert len(self.generator_names) == len(self.generators), "Wrong number of generator names."
        if self.generators_type == GeneratorType.PERMUTATION:
            assert len(self.generators_permutations) > 0
            assert len(self.generators_matrices) == 0
            n = self.state_size
            assert all(len(p) == n for p in self.generators_permutations)
            assert min(self.central_state) >= 0
            assert max(self.central_state) < n
        elif self.generators_type == GeneratorType.MATRIX:
            assert len(self.generators_permutations) == 0
            assert len(self.generators_matrices) > 0
            n = self.generators_matrices[0].matrix.shape[0]
            assert all(g.matrix.shape == (n, n) for g in self.generators_matrices)
            m = self.state_size // n
            assert self.state_size == n * m, "State size must be multiple of generator matrix size."
        else:
            raise ValueError(f"Unknown generator type: {self.generators_type}")

    @cached_property
    def generators(self) -> Union[list[list[int]], list[MatrixGenerator]]:
        if self.generators_type == GeneratorType.PERMUTATION:
            return self.generators_permutations
        else:
            return self.generators_matrices

    @cached_property
    def n_generators(self) -> int:
        return len(self.generators)

    @cached_property
    def state_size(self) -> int:
        return len(self.central_state)

    @cached_property
    def generators_inverse_closed(self) -> bool:
        """Whether for each generator its inverse is also a generator."""
        if self.generators_type == GeneratorType.PERMUTATION:
            generators_set = set(tuple(perm) for perm in self.generators_permutations)
            return all(tuple(inverse_permutation(p)) in generators_set for p in self.generators_permutations)
        else:
            return all(any(g1.is_inverse_to(g2) for g2 in self.generators_matrices) for g1 in self.generators_matrices)

    @cached_property
    def decoded_state_shape(self) -> tuple[int, ...]:
        """Shape of state when presented in decoded (human-readable) format."""
        if self.generators_type == GeneratorType.PERMUTATION:
            return (self.state_size,)
        else:
            n = self.generators_matrices[0].n
            m = self.state_size // n
            assert self.state_size == n * m
            return n, m

    @staticmethod
    def normalize_central_state(central_state: Union[list[int], torch.Tensor, np.ndarray, str]) -> list[int]:
        if hasattr(central_state, "reshape"):
            central_state = central_state.reshape((-1,))  # Flatten.
        return [int(x) for x in central_state]

    def with_central_state(self, central_state) -> "CayleyGraphDef":
        return CayleyGraphDef(
            self.generators_type,
            self.generators_permutations,
            self.generators_matrices,
            self.generator_names,
            CayleyGraphDef.normalize_central_state(central_state),
        )

    def is_permutation_group(self):
        """Whether generators in this graph are permutations."""
        return self.generators_type == GeneratorType.PERMUTATION

    def is_matrix_group(self):
        """Whether generators in this graph are matrices."""
        return self.generators_type == GeneratorType.MATRIX
