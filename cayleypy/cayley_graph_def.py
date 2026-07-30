from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import Optional, Union, Any

import numpy as np
import torch

from .permutation_utils import inverse_permutation

AnyStateType = Union[torch.Tensor, np.ndarray, list]


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
        mx = torch.tensor(self.matrix, dtype=states.dtype, device=states.device)
        mx = mx.unsqueeze(0).unsqueeze(-1)
        ans = (mx * states.unsqueeze(1)).sum(dim=2)
        if self.modulo > 0:
            ans %= self.modulo
        return ans

    @staticmethod
    def _mod_inv(a: int, m: int) -> int:
        """Modular inverse of a modulo m via extended Euclidean algorithm."""
        if m == 1:
            return 0
        g, x, _ = MatrixGenerator._ext_gcd(a % m, m)
        if g != 1:
            raise ValueError(f"{a} is not invertible mod {m}.")
        return x % m

    @staticmethod
    def _ext_gcd(a: int, b: int):
        """Extended Euclidean algorithm returning (g, x, y) with g = a*x + b*y."""
        if b == 0:
            return a, 1, 0
        g, x1, y1 = MatrixGenerator._ext_gcd(b, a % b)
        return g, y1, x1 - (a // b) * y1

    @staticmethod
    def _adjugate(mx: np.ndarray) -> np.ndarray:
        """Adjugate (adjoint) of a square integer matrix.

        For n <= 3 uses explicit formulas; otherwise computes via minors.
        adj(A)[i][j] = (-1)^{i+j} * det(minor(j, i)).
        """
        n = mx.shape[0]
        if n == 1:
            return np.array([[1]], dtype=np.int64)
        if n == 2:
            return np.array([[mx[1, 1], -mx[0, 1]], [-mx[1, 0], mx[0, 0]]], dtype=np.int64)
        if n == 3:
            a, b, c = mx[0, 0], mx[0, 1], mx[0, 2]
            d, e, f = mx[1, 0], mx[1, 1], mx[1, 2]
            g, h, i = mx[2, 0], mx[2, 1], mx[2, 2]
            return np.array(
                [
                    [e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d],
                ],
                dtype=np.int64,
            )
        adj = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(mx, j, axis=0), i, axis=1)
                det_minor = round(np.linalg.det(minor))
                adj[i, j] = ((-1) ** (i + j)) * det_minor
        return adj

    @cached_property
    def inv(self):
        """Inverse of this matrix. Throws error if matrix is not invertible."""
        if self.modulo > 0:
            det = round(np.linalg.det(self.matrix))
            det_inv = MatrixGenerator._mod_inv(det % self.modulo, self.modulo)
            adj = MatrixGenerator._adjugate(self.matrix)
            matrix_inv = (det_inv * adj) % self.modulo
            assert np.array_equal(self.apply(matrix_inv), np.eye(self.n)), "Matrix is not invertible (mod m)."
            return MatrixGenerator.create(matrix_inv, self.modulo)
        matrix_inv = np.array(np.linalg.inv(self.matrix), dtype=np.int64)
        assert np.array_equal(self.apply(matrix_inv), np.eye(self.n)), "Matrix is not invertible."
        return MatrixGenerator.create(matrix_inv, self.modulo)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MatrixGenerator):
            return False
        return self.modulo == other.modulo and np.array_equal(self.matrix, other.matrix)


@dataclass(frozen=True)
class CayleyGraphDef:
    """Mathematical definition of a CayleyGraph."""

    generators_type: GeneratorType
    generators_permutations: list[list[int]]
    generators_matrices: list[MatrixGenerator]
    generator_names: list[str]
    central_state: list[int]
    name: str

    @staticmethod
    def create(
        generators: Union[list[list[int]], torch.Tensor, np.ndarray],
        generator_names: Optional[list[str]] = None,
        central_state: Optional[AnyStateType] = None,
        name: str = "",
    ):
        """Creates Cayley Graph definition (when generators are permutations).

        :param generators: List of generating permutations of size n.
        :param generator_names: Names of the generators (optional).
        :param central_state: List of n numbers between 0 and n-1, the central state.
                 If None, defaults to the identity permutation of size n.
        :param name: Name of this graph.
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

        return CayleyGraphDef(GeneratorType.PERMUTATION, generators_list, [], generator_names, central_state, name)

    @staticmethod
    def for_matrix_group(
        *,
        generators: list[MatrixGenerator],
        generator_names: Optional[list[str]] = None,
        central_state: Optional[AnyStateType] = None,
        name: str = "",
    ):
        """Creates Cayley Graph definition (when generators are matrices).

        :param generators: List of generating n*n matrices.
        :param generator_names: Names of the generators (optional).
        :param central_state: the central state (n*m matrix). Defaults to the n*n identity matrix.
        :param name: Name of this graph.
        """
        if generator_names is None:
            generator_names = ["g" + str(i) for i in range(len(generators))]
        if central_state is None:
            # By default, central element is the identity matrix.
            central_state = np.eye(generators[0].n, dtype=np.int64)
        central_state_list = CayleyGraphDef.normalize_central_state(central_state)
        n = generators[0].n
        assert len(central_state) % n == 0, "Wrong size of central state."
        return CayleyGraphDef(GeneratorType.MATRIX, [], generators, generator_names, central_state_list, name)

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
    def generators_inverse_map(self) -> Optional[list[int]]:
        """Maps generators to their inverses. Returns None if generators are not inverse-closed."""
        ans = []
        if self.generators_type == GeneratorType.PERMUTATION:
            generators_idx = {tuple(self.generators_permutations[i]): i for i in range(self.n_generators)}
            for i in range(self.n_generators):
                inv_perm = tuple(inverse_permutation(self.generators_permutations[i]))
                if inv_perm not in generators_idx:
                    return None
                ans.append(generators_idx[inv_perm])
        else:
            assert self.generators_type == GeneratorType.MATRIX
            for i in range(self.n_generators):
                i_inv = -1
                for j in range(self.n_generators):
                    if self.generators_matrices[i].is_inverse_to(self.generators_matrices[j]):
                        i_inv = j
                if i_inv == -1:
                    return None
                ans.append(i_inv)
        return ans

    @cached_property
    def generators_inverse_closed(self) -> bool:
        """Whether for each generator its inverse is also a generator."""
        return self.generators_inverse_map is not None

    @cached_property
    def decoded_state_shape(self) -> tuple[int, ...]:
        """Shape of state when presented in decoded (human-readable) format."""
        if self.generators_type == GeneratorType.PERMUTATION:
            return (self.state_size,)
        else:
            assert self.generators_type == GeneratorType.MATRIX
            n = self.generators_matrices[0].n
            m = self.state_size // n
            assert self.state_size == n * m
            return n, m

    @staticmethod
    def normalize_central_state(
        central_state: Union[list[int], list[list[int]], torch.Tensor, np.ndarray, str],
    ) -> list[int]:
        if isinstance(central_state, list):
            central_state = np.array(central_state)
        if hasattr(central_state, "reshape"):
            central_state = central_state.reshape((-1,))  # Flatten.
        return [int(x) for x in central_state]

    def with_central_state(self, central_state) -> "CayleyGraphDef":
        return replace(self, central_state=CayleyGraphDef.normalize_central_state(central_state))

    def with_name(self, name: str) -> "CayleyGraphDef":
        return replace(self, name=name)

    def is_permutation_group(self):
        """Whether generators in this graph are permutations."""
        return self.generators_type == GeneratorType.PERMUTATION

    def is_matrix_group(self):
        """Whether generators in this graph are matrices."""
        return self.generators_type == GeneratorType.MATRIX

    def with_inverted_generators(self) -> "CayleyGraphDef":
        """Returns the same graph where generators are replaced with inverses (in the same order).

        This is needed for restoring path in the Beam Search algorithm.
        Note that even when generators are self-inverse, this will be a different graph because order of generators
        changes. For example, LRX generators turn into RLX generators.
        """
        if self.generators_type == GeneratorType.PERMUTATION:
            return CayleyGraphDef.create(
                generators=[inverse_permutation(p) for p in self.generators_permutations],
                central_state=self.central_state,
            )
        else:
            assert self.generators_type == GeneratorType.MATRIX
            return CayleyGraphDef.for_matrix_group(
                generators=[m.inv for m in self.generators_matrices],
                central_state=self.central_state,
            )

    def make_inverse_closed(self) -> "CayleyGraphDef":
        """Makes generators inverse-closed, adding extra generators when necessary.

        If generators are already inverse-closed, returns self.
        Otherwise, for each generator that does not have its inverse in the set of generators, adds an inverse generator
        to the set.
        """
        if self.generators_inverse_closed:
            return self

        new_name = self.name
        if new_name != "":
            new_name += "-ic"
        new_generator_names = []  # type: list[str]

        if self.generators_type == GeneratorType.PERMUTATION:
            generators_set = {tuple(self.generators_permutations[i]) for i in range(self.n_generators)}
            new_generators_permutations = []
            for i in range(self.n_generators):
                inv_perm = inverse_permutation(self.generators_permutations[i])
                if tuple(inv_perm) not in generators_set:
                    new_generators_permutations.append(inv_perm)
                    new_generator_names.append(self.generator_names[i] + "'")
            return CayleyGraphDef.create(
                generators=self.generators_permutations + new_generators_permutations,
                generator_names=self.generator_names + new_generator_names,
                central_state=self.central_state,
                name=new_name,
            )
        else:
            assert self.generators_type == GeneratorType.MATRIX
            new_generators_matrices = []
            for i in range(self.n_generators):
                has_inverse = any(
                    self.generators_matrices[i].is_inverse_to(self.generators_matrices[j])
                    for j in range(self.n_generators)
                )
                if not has_inverse:
                    new_generators_matrices.append(self.generators_matrices[i].inv)
                    new_generator_names.append(self.generator_names[i] + "'")
            return CayleyGraphDef.for_matrix_group(
                generators=self.generators_matrices + new_generators_matrices,
                generator_names=self.generator_names + new_generator_names,
                central_state=self.central_state,
                name=new_name,
            )

    def path_to_string(self, path: list[int], delimiter=".") -> str:
        return delimiter.join(self.generator_names[i] for i in path)

    def revert_path(self, path: list[int]) -> list[int]:
        """Given path A->B, returns path B->A. Only for inverse-closed generators."""
        idx = self.generators_inverse_map
        assert idx is not None, "Cannot revert path because generators are not inverse closed."
        return [idx[i] for i in path[::-1]]
