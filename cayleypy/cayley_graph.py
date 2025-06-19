import functools
import gc
import math
from typing import Callable, Optional, Union

import numpy as np
import torch

from .bfs_result import BfsResult
from .hasher import StateHasher
from .permutation_utils import *
from .string_encoder import StringEncoder
from .torch_utils import isin_via_searchsorted


class CayleyGraph:
    """Represents a Schreier coset graph for the group S_n (group of n-element permutations).

    In this graph:
      * Vertices (aka "states") are strings of integers of size n.
      * Edges are permutations of size n from given set of `generators`.
      * There is an outgoing edge for every vertex A and every generating permutation P.
      * On the other end of this edge, there is a vertex P(A).
    In general case, this graph is directed. However, in the case when set of generators is closed under inversion,
        every edge has and edge in other direction, so the graph can be viewed as undirected.
    The graph is fully defined by list of generators and one selected state called "destination state". It contains
        all vertices reachable from the destination state.
    In the case when destination state is a permutation itself, and generators fully generate S_n, this is a Cayley
        graph for S_n, hence the name. In more general case, elements can have less than n distinct values, and we call
        the set of vertices "coset".
    """

    def __init__(
            self,
            generators: Union[list[list[int]], torch.Tensor, np.ndarray],
            *,
            generator_names: Optional[list[str]] = None,
            dest: Union[list[int], torch.Tensor, np.ndarray, str, None] = None,
            device: str = "auto",
            random_seed: Optional[int] = None,
            bit_encoding_width: Union[Optional[int], str] = "auto",
            verbose: int = 0,
            batch_size: int = 2 ** 25,
            hash_chunk_size: int = 2 ** 25,
            memory_limit_gb: float = 16):
        """Initializes CayleyGraph.

        :param generators: List of generating permutations of size n.
        :param generators: Names of the generators (optional).
        :param dest: List of n numbers between 0 and n-1, the destination state.
                 If None, defaults to the identity permutation of size n.
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
        self.verbose = verbose
        self.batch_size = batch_size
        self.memory_limit_bytes = int(memory_limit_gb * (2 ** 30))

        # Pick device. It will be used to store all tensors.
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if verbose > 0:
            print(f"Using device: {self.device}.")

        # Prepare generators.
        if isinstance(generators, list):
            generators_list = generators
        elif isinstance(generators, torch.Tensor):
            generators_list = [[q.item() for q in generators[i, :]] for i in range(generators.shape[0])]
        elif isinstance(generators, np.ndarray):
            generators_list = [list(generators[i, :]) for i in range(generators.shape[0])]
        else:
            raise ValueError('Unsupported format for "generators" ' + str(type(generators)))
        self.generators = torch.tensor(generators_list, dtype=torch.int64, device=self.device)

        # Validate generators.
        self.state_size = len(generators_list[0])  # Size of permutations.
        self.n_generators = len(generators_list)
        generators_set = set(tuple(perm) for perm in generators_list)
        id_perm = list(range(self.state_size))
        self.generators_inverse_closed = True
        for perm in generators_list:
            assert sorted(perm) == id_perm, f"{perm} is not a permutation of length {self.state_size}."
            if tuple(inverse_permutation(perm)) not in generators_set:
                self.generators_inverse_closed = False

        # Prepare destination state.
        if dest is None:
            dest = list(range(self.state_size))  # Identity permutation.
        elif type(dest) is str:
            dest = [int(x) for x in dest]
        self.destination_state = torch.as_tensor(dest, device=self.device, dtype=torch.int64)
        assert self.destination_state.shape == (self.state_size,)
        assert int(self.destination_state.min()) >= 0
        assert int(self.destination_state.max()) < self.state_size

        # Prepare encoder in case we want to encode states using few bits per element.
        if bit_encoding_width == "auto":
            bit_encoding_width = int(math.ceil(math.log2(int(self.destination_state.max()) + 1)))
        self.string_encoder: Optional[StringEncoder] = None
        encoded_state_size: int = self.state_size
        if bit_encoding_width is not None:
            self.string_encoder = StringEncoder(code_width=int(bit_encoding_width), n=self.state_size)
            self.encoded_generators = [self.string_encoder.implement_permutation(perm) for perm in generators]
            encoded_state_size = self.string_encoder.encoded_length

        self.hasher = StateHasher(encoded_state_size, random_seed, self.device, chunk_size=hash_chunk_size)

        # Prepare generator names.
        if generator_names is not None:
            self.generator_names = generator_names
        else:
            self.generator_names = [",".join(str(int(i)) for i in g) for g in self.generators]

    def get_unique_states(self,
                          states: torch.Tensor,
                          hashes: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states`. May change order."""
        if hashes is None:
            hashes = self.hasher.make_hashes(states)
        hashes_sorted, idx = torch.sort(hashes, stable=True)

        # Compute mask of first occurrences for each unique value.
        mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=self.device)
        if hashes_sorted.size(0) > 1:
            mask[1:] = (hashes_sorted[1:] != hashes_sorted[:-1])

        unique_idx = idx[mask]
        unique_states = states[unique_idx]
        unique_hashes = self.hasher.make_hashes(unique_states) if self.hasher.is_identity else hashes[unique_idx]
        return unique_states, unique_hashes, unique_idx

    def _encode_states(self, states: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        states = torch.as_tensor(states, device=self.device)
        if len(states.shape) == 1:  # In case when only one state was passed.
            states = states.reshape(1, -1)
        assert len(states.shape) == 2
        assert states.shape[1] == self.state_size
        if self.string_encoder is not None:
            return self.string_encoder.encode(states)
        return states

    def _decode_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.string_encoder is not None:
            return self.string_encoder.decode(states)
        return states

    def _get_neighbors(self, states: torch.Tensor, dest: torch.Tensor):
        """Calculates all neighbors of `states`, writes them to `dest`, which must be initialized to zeros."""
        states_num = states.shape[0]
        assert dest.shape[0] == states_num * self.n_generators
        if self.string_encoder is not None:
            for i in range(self.n_generators):
                self.encoded_generators[i](states, dest[i * states_num:(i + 1) * states_num])
        else:
            moves = self.generators
            dest[:, :] = torch.gather(
                states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
                2,
                moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1))
            ).flatten(end_dim=1)

    def _get_neighbors_batched(self, states: torch.Tensor) -> torch.Tensor:
        estimated_result_size = states.shape[0] * states.shape[1] * self.n_generators * 8
        if 5 * estimated_result_size > self.memory_limit_bytes:
            self._free_memory()

        if self.string_encoder is not None or states.shape[0] <= self.batch_size:
            neighbors = torch.zeros((states.shape[0] * self.n_generators, states.shape[1]), dtype=torch.int64,
                                    device=self.device)
            self._get_neighbors(states, neighbors)
        else:
            num_batches = int(math.ceil(states.shape[0] / self.batch_size))
            neighbors = torch.zeros((self.n_generators * states.shape[0], states.shape[1]), dtype=torch.int64,
                                    device=self.device)
            i = 0
            for batch in states.tensor_split(num_batches, dim=0):
                num_neighbors = self.n_generators * batch.shape[0]
                self._get_neighbors(batch, neighbors[i:i + num_neighbors, :])
                i += num_neighbors
        return neighbors

    def bfs(self,
            *,
            start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
            max_layer_size_to_store: Optional[int] = 1000,
            max_layer_size_to_explore: int = 10 ** 9,
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
        assert self.generators_inverse_closed, "BFS is supported only when generators are inverse-closed."

        start_states = self._encode_states(start_states or self.destination_state)
        layer0_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
        layer1, layer1_hashes, _ = self.get_unique_states(start_states)
        layer_sizes = [len(layer1)]
        layers = {0: self._decode_states(layer1)}
        full_graph_explored = False
        edges_list_starts = []
        edges_list_ends = []
        all_layers_hashes = []
        max_layer_size_to_store = max_layer_size_to_store or 10 ** 15

        for i in range(1, max_diameter + 1):
            layer1_neighbors = self._get_neighbors_batched(layer1)
            layer1_neighbors_hashes = self.hasher.make_hashes(layer1_neighbors)
            if return_all_edges:
                if self.string_encoder is not None:
                    edges_list_starts += [layer1_hashes] * self.n_generators
                else:
                    edges_list_starts.append(layer1_hashes.repeat_interleave(self.n_generators))
                edges_list_ends.append(layer1_neighbors_hashes)
            if return_all_hashes:
                all_layers_hashes.append(layer1_hashes)

            # BFS iteration: layer2 := neighbors(layer1)-layer0-layer1.
            layer2, layer2_hashes, _ = self.get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)
            mask = ~isin_via_searchsorted(layer2_hashes, layer1_hashes)
            if i > 1:
                mask &= ~isin_via_searchsorted(layer2_hashes, layer0_hashes)
            layer2 = layer2[mask]
            layer2_hashes = self.hasher.make_hashes(layer2) if self.hasher.is_identity else layer2_hashes[mask]

            if len(layer2) == 0:
                full_graph_explored = True
                break
            if self.verbose >= 2:
                print(f"Layer {i}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            if len(layer2) <= max_layer_size_to_store:
                layers[i] = self._decode_states(layer2)

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

        layers[len(layer_sizes) - 1] = self._decode_states(layer1)

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=layers,
            bfs_completed=full_graph_explored,
            vertices_hashes=vertices_hashes,
            edges_list_hashes=edges_list_hashes,
            graph=self)

    @functools.cache
    def to_networkx_graph(self):
        return self.bfs(max_layer_size_to_store=10 ** 18, return_all_edges=True,
                        return_all_hashes=True).to_networkx_graph()

    def _free_memory(self):
        if self.verbose >= 1:
            print("Freeing memory...")
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
