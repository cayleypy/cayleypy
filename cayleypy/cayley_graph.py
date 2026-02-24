import gc
import math
from functools import cached_property
from typing import Optional, Union

import torch

from cayleypy.algo.bfs_result import BfsResult
from .algo.beam_search import BeamSearchAlgorithm
from .algo.bfs_algo import BfsAlgorithm
from .algo.random_walks import RandomWalksGenerator
from .cayley_graph_def import AnyStateType, CayleyGraphDef, GeneratorType
from .hasher import StateHasher
from .string_encoder import StringEncoder
from .torch_utils import isin_via_searchsorted


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
    all vertices reachable from the central state. This definition is encapsulated in :class:`cayleypy.CayleyGraphDef`.

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
        _hasher: Optional[StateHasher] = None,
        **unused_kwargs,
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
        self.bit_encoding_width = bit_encoding_width

        # Pick device. It will be used to store all tensors.
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if verbose > 0:
            print(f"Using device: {self.device}.")

        self.central_state = torch.as_tensor(definition.central_state, device=self.device, dtype=torch.int64)
        self.encoded_state_size: int = self.definition.state_size
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
                self.encoded_state_size = self.string_encoder.encoded_length

        if _hasher is not None:
            self.hasher = _hasher
        else:
            self.hasher = StateHasher(self, random_seed, chunk_size=hash_chunk_size)
        self.central_state_hash = self.hasher.make_hashes(self.encode_states(self.central_state))

    def get_unique_states(
        self, states: torch.Tensor, hashes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states` and sorts them by hash."""
        if self.hasher.is_identity:
            unique_hashes = torch.unique(states.reshape(-1), sorted=True)
            return unique_hashes.reshape((-1, 1)), unique_hashes
        if hashes is None:
            hashes = self.hasher.make_hashes(states)
        hashes_sorted, idx = torch.sort(hashes, stable=True)

        # Compute mask of first occurrences for each unique value.
        mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=self.device)
        if hashes_sorted.size(0) > 1:
            mask[1:] = hashes_sorted[1:] != hashes_sorted[:-1]

        unique_idx = idx[mask]
        return states[unique_idx], hashes[unique_idx]

    def encode_states(self, states: AnyStateType) -> torch.Tensor:
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

    def apply_generator_batched(self, i: int, src: torch.Tensor, dst: torch.Tensor):
        """Applies i-th generator to encoded states in `src`, writes output to `dst`."""
        states_num = src.shape[0]
        if self.definition.is_permutation_group():
            if self.string_encoder is not None:
                self.encoded_generators[i](src, dst)
            else:
                move = self.permutations_torch[i].reshape((1, -1)).expand(states_num, -1)
                dst[:, :] = torch.gather(src, 1, move)
        else:
            assert self.definition.is_matrix_group()
            n, m = self.definition.decoded_state_shape
            mx = self.definition.generators_matrices[i]
            src = src.reshape((states_num, n, m))
            dst[:, :] = mx.apply_batch_torch(src).reshape((states_num, n * m))

    def apply_path(self, states: AnyStateType, generator_ids: list[int]) -> torch.Tensor:
        """Applies multiple generators to given state(s) in order.

        :param states: one or more states (as torch.Tensor) to which to apply the states.
        :param generator_ids: Indexes of generators to apply.
        :return: States after applying specified generators in order.
        """
        states = self.encode_states(states)
        for gen_id in generator_ids:
            assert 0 <= gen_id < self.definition.n_generators
            new_states = torch.zeros_like(states)
            self.apply_generator_batched(gen_id, states, new_states)
            states = new_states
        return self.decode_states(states)

    def validate_path(self, start_state: AnyStateType, path: list[int]):
        """Checks that `path` indeed is path from `start_state` to central state."""
        path_result = self.apply_path(start_state, path).reshape(-1)
        assert torch.equal(path_result, self.central_state)

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates all neighbors of `states` (in internal representation)."""
        states_num = states.shape[0]
        neighbors = torch.zeros(
            (states_num * self.definition.n_generators, states.shape[1]), dtype=torch.int64, device=self.device
        )
        for i in range(self.definition.n_generators):
            dst = neighbors[i * states_num : (i + 1) * states_num, :]
            self.apply_generator_batched(i, states, dst)
        return neighbors

    def get_neighbors_decoded(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates neighbors in decoded (external) representation."""
        return self.decode_states(self.get_neighbors(self.encode_states(states)))

    def bfs(self, **kwargs) -> BfsResult:
        """Runs bread-first search (BFS) algorithm from given `start_states`.

        See :class:`cayleypy.algo.BfsAlgorithm` for more details, including arguments description.
        """
        return BfsAlgorithm.bfs(self, **kwargs)

    def random_walks(self, **kwargs):
        """Generates random walks on this graph.

        See :class:`cayleypy.algo.RandomWalksGenerator` for more details.
        """
        return RandomWalksGenerator(self).generate(**kwargs)

    def beam_search(self, **kwargs):
        """Tries to find a path from `start_state` to central state using Beam Search algorithm.

        See :class:`cayleypy.algo.BeamSearchAlgorithm` for more details.
        """
        return BeamSearchAlgorithm(self).search(**kwargs)

    def restore_path(self, hashes: list[torch.Tensor], to_state: AnyStateType) -> list[int]:
        """Restores path from layers hashes.

        Layers must be such that there is edge from state on previous layer to state on next layer.
        The end of the path is to_state.
        Last layer in `hashes` must contain a state from which there is a transition to `to_state`.
        `to_state` must be in "decoded" format.
        Length of returned path is equal to number of layers.
        """
        inv_graph = self.with_inverted_generators
        path = []  # type: list[int]
        cur_state = self.decode_states(self.encode_states(to_state))

        for i in range(len(hashes) - 1, -1, -1):
            # Find hash in hashes[i] from which we could go to cur_state.
            # Corresponding state will be new_cur_state.
            # The generator index in inv_graph that moves cur_state->new_cur_state is the same as generator index
            # in this graph that moves new_cur_state->cur_state - this is what we append to the answer.
            candidates = inv_graph.get_neighbors_decoded(cur_state)
            candidates_hashes = self.hasher.make_hashes(self.encode_states(candidates))
            mask = torch.isin(candidates_hashes, hashes[i])
            assert torch.any(mask), "Not found any neighbor on previous layer."
            gen_id = int(mask.nonzero()[0].item())
            path.append(gen_id)
            cur_state = candidates[gen_id : gen_id + 1, :]
        return path[::-1]

    def find_path_to(self, end_state: AnyStateType, bfs_result: BfsResult) -> Optional[list[int]]:
        """Finds path from central_state to end_state using pre-computed BfsResult.

        :param end_state: Final state of the path.
        :param bfs_result: Pre-computed BFS result (call `bfs(return_all_hashes=True)` to get this).
        :return: The found path (list of generator ids), or None if `end_state` is not reachable from `start_state`.
        """
        assert bfs_result.graph == self.definition
        end_state_hash = self.hasher.make_hashes(self.encode_states(end_state))
        bfs_result.check_has_layer_hashes()
        layers_hashes = bfs_result.layers_hashes
        for i, bfs_layer in enumerate(layers_hashes):
            if bool(isin_via_searchsorted(end_state_hash, bfs_layer)):
                return self.restore_path(layers_hashes[:i], end_state)
        return None

    def find_path_from(self, start_state: AnyStateType, bfs_result: BfsResult) -> Optional[list[int]]:
        """Finds path from start_state to central_state using pre-computed BfsResult.

        This is possible only for inverse-closed generators.

        :param start_state: First state of the path.
        :param bfs_result: Pre-computed BFS result (call `bfs(return_all_hashes=True)` to get this).
        :return: The found path (list of generator ids), or None if central_state is not reachable from start_state.
        """
        assert self.definition.generators_inverse_closed
        path_to = self.find_path_to(start_state, bfs_result)
        if path_to is None:
            return None
        return self.definition.revert_path(path_to)

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

    @cached_property
    def with_inverted_generators(self):
        """Returns copy of this graph with inverted generators."""
        return self.modified_copy(self.definition.with_inverted_generators())

    def modified_copy(self, new_def: CayleyGraphDef) -> "CayleyGraph":
        """Makes a copy of this graph with different definition but other parameters unchanged.

        The new graph will use the same encoding and hashing for states as the original.
        """
        ans = CayleyGraph(
            new_def,
            _hasher=self.hasher,
            bit_encoding_width=self.bit_encoding_width,
        )
        ans.hasher = self.hasher
        ans.string_encoder = self.string_encoder
        return ans
