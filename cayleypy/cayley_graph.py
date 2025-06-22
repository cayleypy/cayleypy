import gc
import math
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Callable, Optional, Union

import numpy as np
import torch

from .bfs_result import BfsResult
from .hasher import StateHasher
from .permutation_utils import inverse_permutation
from .string_encoder import StringEncoder
from .torch_utils import isin_via_searchsorted


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
        ans = torch.einsum("ij,bjk->bik", mx, states)
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


class CayleyGraph:
    """Represents a Schreier coset graph for some group.

    In this graph:
      * Vertices (aka "states") are integer vectors or matrices.
      * There is an outgoing edge for every vertex A and every generator G.
      * On the other end of this edge, there is a vertex G(A).
    When `definition.generator_type` is `PERMUTATION`:
      * The group is the group of permutations S_n.
      * Generators are permutations of n elements.
      * States are vectors of integers of size n.
    When `definition.generator_type` is `MATRIX`:
      * The group is the group of n*n integer matrices under multiplication (usual or modular)
      * Technically, it's a group only when all generators are invertible, but we don't require this.
      * Generators are n*n integer matrices.
      * States are n*m integers matrices.
    In general case, this graph is directed. However, in the case when set of generators is closed under inversion,
        every edge has and edge in other direction, so the graph can be viewed as undirected.
    The graph is fully defined by list of generators and one selected state called "central state". The graph contains
        all vertices reachable from the central state. This definition is encapsulated in CayleyGraphDef,
    In the case when the central state is a permutation itself, and generators fully generate S_n, this is a Cayley
        graph, hence the name. In more general case, elements can have less than n distinct values, and we call
        the set of vertices "coset".
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
    ):
        """Initializes CayleyGraph.

        :param definition: definition of the graph (as CayleyPyDef).
        :param device: one of ['auto','cpu','cuda'] - PyTorch device to store all tensors.
        :param random_seed: random seed for deterministic hashing.
        :param bit_encoding_width: how many bits (between 1 and 63) to use to encode one element in a state.
                 If 'auto', optimal width will be picked.
                 If None, elements will be encoded by int64 numbers.
        :param verbose: Level of logging. 0 means no logging.
        :param batch_size: Size of batch for batch processing.
        :param hash_chunk_size: Size of chunk for hashing.
        :param memory_limit_gb: Approximate available memory, in GB.
                 It is safe to set this to less than available on your machine, it will just cause more frequent calls
                 to the "free memory" function.
        """
        self.definition = definition
        self.verbose = verbose
        self.batch_size = batch_size
        self.memory_limit_bytes = int(memory_limit_gb * (2**30))

        # Pick device. It will be used to store all tensors.
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if verbose > 0:
            print(f"Using device: {self.device}.")

        self.central_state = torch.as_tensor(definition.central_state, device=self.device, dtype=torch.int64)
        encoded_state_size: int = self.definition.state_size
        self.string_encoder: Optional[StringEncoder] = None

        if definition.is_permutation_group():
            self.permutations_torch = torch.tensor(
                definition.generators_permutations, dtype=torch.int64, device=self.device
            )

            # Prepare encoder in case we want to encode states using few bits per element.
            if bit_encoding_width == "auto":
                bit_encoding_width = int(math.ceil(math.log2(int(self.central_state.max()) + 1)))
            if bit_encoding_width is not None:
                self.string_encoder = StringEncoder(code_width=int(bit_encoding_width), n=self.definition.state_size)
                self.encoded_generators = [
                    self.string_encoder.implement_permutation(perm) for perm in definition.generators_permutations
                ]
                encoded_state_size = self.string_encoder.encoded_length

        self.hasher = StateHasher(encoded_state_size, random_seed, self.device, chunk_size=hash_chunk_size)

    def get_unique_states(
        self, states: torch.Tensor, hashes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states`. May change order."""
        if hashes is None:
            hashes = self.hasher.make_hashes(states)
        hashes_sorted, idx = torch.sort(hashes, stable=True)

        # Compute mask of first occurrences for each unique value.
        mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=self.device)
        if hashes_sorted.size(0) > 1:
            mask[1:] = hashes_sorted[1:] != hashes_sorted[:-1]

        unique_idx = idx[mask]
        unique_states = states[unique_idx]
        unique_hashes = self.hasher.make_hashes(unique_states) if self.hasher.is_identity else hashes[unique_idx]
        return unique_states, unique_hashes, unique_idx

    def encode_states(self, states: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """Converts states from human-readable to internal representation."""
        states = torch.as_tensor(states, device=self.device)
        states = states.reshape((-1, self.definition.state_size))
        if self.string_encoder is not None:
            return self.string_encoder.encode(states)
        return states

    def decode_states(self, states: torch.Tensor) -> torch.Tensor:
        """Converts states from internal to human-readable representation."""
        if self.definition.generators_type == GeneratorType.MATRIX:
            n, m = self.definition.decoded_state_shape
            # Internally states are vectors, but mathematically they are n*m matrices.
            return states.reshape((-1, n, m))
        if self.string_encoder is not None:
            return self.string_encoder.decode(states)
        return states

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates all neighbors of `states` (in internal representation)."""
        states_num = states.shape[0]
        neighbors = torch.zeros(
            (states_num * self.definition.n_generators, states.shape[1]), dtype=torch.int64, device=self.device
        )
        if self.definition.is_permutation_group():
            if self.string_encoder is not None:
                for i in range(self.definition.n_generators):
                    self.encoded_generators[i](states, neighbors[i * states_num : (i + 1) * states_num])
            else:
                moves = self.permutations_torch
                neighbors[:, :] = torch.gather(
                    states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
                    2,
                    moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1)),
                ).flatten(end_dim=1)
        else:
            assert self.definition.generators_type == GeneratorType.MATRIX
            n, m = self.definition.decoded_state_shape
            states = states.reshape((states_num, n, m))
            for i, mx in enumerate(self.definition.generators_matrices):
                nb = mx.apply_batch_torch(states).reshape((states_num, n * m))
                neighbors[i * states_num : (i + 1) * states_num] = nb

        return neighbors

    def bfs(
        self,
        *,
        start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**9,
        max_diameter: int = 1000000,
        return_all_edges: bool = False,
        return_all_hashes: bool = False,
        keep_alive_func: Callable[[], None] = lambda: None,
    ) -> BfsResult:
        """Runs bread-first search (BFS) algorithm from given `start_states`.

        BFS visits all vertices of the graph in layers, where next layer contains vertices adjacent to previous layer
        that were not visited before. As a result, we get all vertices grouped by their distance from the set of initial
        states.

        Depending on parameters below, it can be used to:
          * Get growth function (number of vertices at each BFS layer).
          * Get vertices at some first and last layers.
          * Get all vertices.
          * Get all vertices and edges (i.e. get the whole graph explicitly).

        :param start_states: states on 0-th layer of BFS. Defaults to destination state of the graph.
        :param max_layer_size_to_store: maximal size of layer to store.
               If None, all layers will be stored (use this if you need full graph).
               Defaults to 1000.
               First and last layers are always stored.
        :param max_layer_size_to_explore: if reaches layer of larger size, will stop the BFS.
        :param max_diameter: maximal number of BFS iterations.
        :param return_all_edges: whether to return list of all edges (uses more memory).
        :param return_all_hashes: whether to return hashes for all vertices (uses more memory).
        :param keep_alive_func - function to call on every iteration.
        :return: BfsResult object with requested BFS results.
        """
        # This version of BFS is correct only for undirected graph.
        assert self.definition.generators_inverse_closed, "BFS is supported only when generators are inverse-closed."

        start_states = self.encode_states(start_states or self.central_state)
        layer0_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
        layer1, layer1_hashes, _ = self.get_unique_states(start_states)
        layer_sizes = [len(layer1)]
        layers = {0: self.decode_states(layer1)}
        full_graph_explored = False
        edges_list_starts = []
        edges_list_ends = []
        all_layers_hashes = []
        max_layer_size_to_store = max_layer_size_to_store or 10**15

        # When state fits in a single int64 and we don't need edges, we can apply more memory-efficient algorithm
        # with batching. This algorithm finds neighbors in batches and removes duplicates from batches before
        # stacking them.
        do_batching = (
            self.string_encoder is not None and self.string_encoder.encoded_length == 1 and not return_all_edges
        )

        # BFS iteration: layer2 := neighbors(layer1)-layer0-layer1.
        for i in range(1, max_diameter + 1):
            if do_batching and len(layer1) > self.batch_size:
                num_batches = int(math.ceil(layer1_hashes.shape[0] / self.batch_size))
                layer2_batches = []  # type: list[torch.Tensor]
                for layer1_batch in layer1.tensor_split(num_batches, dim=0):
                    layer2_batch = self.get_neighbors(layer1_batch).reshape((-1,))
                    layer2_batch = torch.unique(layer2_batch, sorted=True)
                    mask = ~isin_via_searchsorted(layer2_batch, layer1_hashes)
                    if i > 1:
                        mask &= ~isin_via_searchsorted(layer2_batch, layer0_hashes)
                    for other_batch in layer2_batches:
                        mask &= ~isin_via_searchsorted(layer2_batch, other_batch)
                    layer2_batch = layer2_batch[mask]
                    if len(layer2_batch) > 0:
                        layer2_batches.append(layer2_batch)
                if len(layer2_batches) == 0:
                    layer2_hashes = torch.empty((0,))
                else:
                    layer2_hashes = torch.hstack(layer2_batches)
                    layer2_hashes, _ = torch.sort(layer2_hashes)
                layer2 = layer2_hashes.reshape((-1, 1))
            else:
                layer1_neighbors = self.get_neighbors(layer1)
                layer1_neighbors_hashes = self.hasher.make_hashes(layer1_neighbors)
                if return_all_edges:
                    if self.string_encoder is None and self.definition.is_permutation_group():
                        edges_list_starts.append(layer1_hashes.repeat_interleave(self.definition.n_generators))
                    else:
                        edges_list_starts += [layer1_hashes] * self.definition.n_generators
                    edges_list_ends.append(layer1_neighbors_hashes)

                layer2, layer2_hashes, _ = self.get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)
                mask = ~isin_via_searchsorted(layer2_hashes, layer1_hashes)
                if i > 1:
                    mask &= ~isin_via_searchsorted(layer2_hashes, layer0_hashes)
                layer2 = layer2[mask]
                layer2_hashes = self.hasher.make_hashes(layer2) if self.hasher.is_identity else layer2_hashes[mask]

            if layer2.shape[0] * layer2.shape[1] * 8 > 0.1 * self.memory_limit_bytes:
                self.free_memory()
            if return_all_hashes:
                all_layers_hashes.append(layer1_hashes)
            if len(layer2) == 0:
                full_graph_explored = True
                break
            if self.verbose >= 2:
                print(f"Layer {i}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            if len(layer2) <= max_layer_size_to_store:
                layers[i] = self.decode_states(layer2)

            layer1 = layer2
            layer0_hashes, layer1_hashes = layer1_hashes, layer2_hashes
            if len(layer2) >= max_layer_size_to_explore:
                break
            keep_alive_func()

        if return_all_hashes and not full_graph_explored:
            all_layers_hashes.append(layer1_hashes)

        if not full_graph_explored and self.verbose > 0:
            print("BFS stopped before graph was fully explored.")

        edges_list_hashes: Optional[torch.Tensor] = None
        if return_all_edges:
            edges_list_hashes = torch.vstack([torch.hstack(edges_list_starts), torch.hstack(edges_list_ends)]).T
        vertices_hashes: Optional[torch.Tensor] = None
        if return_all_hashes:
            vertices_hashes = torch.hstack(all_layers_hashes)

        layers[len(layer_sizes) - 1] = self.decode_states(layer1)

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=layers,
            bfs_completed=full_graph_explored,
            vertices_hashes=vertices_hashes,
            edges_list_hashes=edges_list_hashes,
            graph=self.definition,
        )

    def to_networkx_graph(self):
        return self.bfs(
            max_layer_size_to_store=10**18, return_all_edges=True, return_all_hashes=True
        ).to_networkx_graph()

    def free_memory(self):
        if self.verbose >= 1:
            print("Freeing memory...")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    @property
    def generators(self):
        """Generators of this Cayley graph."""
        return self.definition.generators
