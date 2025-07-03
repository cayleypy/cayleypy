import gc
import math
from typing import Optional, Union

import numpy as np
import torch

from .beam_search_result import BeamSearchResult
from .bfs_result import BfsResult
from .cayley_graph_def import CayleyGraphDef, GeneratorType
from .hasher import StateHasher
from .predictor import Predictor
from .string_encoder import StringEncoder
from .torch_utils import isin_via_searchsorted, TorchHashSet


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

        self.hasher = StateHasher(self.encoded_state_size, random_seed, self.device, chunk_size=hash_chunk_size)
        self.central_state_hash = self.hasher.make_hashes(self.encode_states(self.central_state))

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

    def _apply_generator_batched(self, i: int, src: torch.Tensor, dst: torch.Tensor):
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

    def apply_path(self, states: torch.Tensor, generator_ids: list[int]) -> torch.Tensor:
        """Applies multiple generators to given state(s) in order.

        :param states: one or more states (as torch.Tensor) to which to apply the states.
        :param generator_ids: Indexes of generators to apply.
        :return: States after applying specified generators in order.
        """
        states = self.encode_states(states)
        for gen_id in generator_ids:
            assert 0 <= gen_id < self.definition.n_generators
            new_states = torch.zeros_like(states)
            self._apply_generator_batched(gen_id, states, new_states)
            states = new_states
        return self.decode_states(states)

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates all neighbors of `states` (in internal representation)."""
        states_num = states.shape[0]
        neighbors = torch.zeros(
            (states_num * self.definition.n_generators, states.shape[1]), dtype=torch.int64, device=self.device
        )
        for i in range(self.definition.n_generators):
            dst = neighbors[i * states_num : (i + 1) * states_num, :]
            self._apply_generator_batched(i, states, dst)
        return neighbors

    def get_neighbors_decoded(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates neighbors in decoded (external) representation."""
        return self.decode_states(self.get_neighbors(self.encode_states(states)))

    def bfs(
        self,
        *,
        start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**9,
        max_diameter: int = 1000000,
        return_all_edges: bool = False,
        return_all_hashes: bool = False,
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

        :return: BfsResult object with requested BFS results.
        """
        start_states = self.encode_states(start_states or self.central_state)
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

        # Stores hashes of previous layers, so BFS does not visit already visited states again.
        # If generators are inverse closed, only 2 last layers are stored here.
        seen_states_hashes = [layer1_hashes]

        # Returns mask where 0s are at positions in `current_layer_hashes` that were seen previously.
        def remove_seen_states(current_layer_hashes: torch.Tensor) -> torch.Tensor:
            ans = ~isin_via_searchsorted(current_layer_hashes, seen_states_hashes[-1])
            for h in seen_states_hashes[:-1]:
                ans &= ~isin_via_searchsorted(current_layer_hashes, h)
            return ans

        # BFS iteration: layer2 := neighbors(layer1)-layer0-layer1.
        for i in range(1, max_diameter + 1):
            if do_batching and len(layer1) > self.batch_size:
                num_batches = int(math.ceil(layer1_hashes.shape[0] / self.batch_size))
                layer2_batches = []  # type: list[torch.Tensor]
                for layer1_batch in layer1.tensor_split(num_batches, dim=0):
                    layer2_batch = self.get_neighbors(layer1_batch).reshape((-1,))
                    layer2_batch = torch.unique(layer2_batch, sorted=True)
                    mask = remove_seen_states(layer2_batch)
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
                    edges_list_starts += [layer1_hashes.repeat(self.definition.n_generators)]
                    edges_list_ends.append(layer1_neighbors_hashes)

                layer2, layer2_hashes, _ = self.get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)
                mask = remove_seen_states(layer2_hashes)
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
            layer1_hashes = layer2_hashes
            seen_states_hashes.append(layer2_hashes)
            if self.definition.generators_inverse_closed:
                # Only keep hashes for last 2 layers.
                seen_states_hashes = seen_states_hashes[-2:]
            if len(layer2) >= max_layer_size_to_explore:
                break

        if return_all_hashes and not full_graph_explored:
            all_layers_hashes.append(layer1_hashes)

        if not full_graph_explored and self.verbose > 0:
            print("BFS stopped before graph was fully explored.")

        edges_list_hashes: Optional[torch.Tensor] = None
        if return_all_edges:
            if not full_graph_explored:
                # Add copy of edges between last 2 layers, but in opposite direction.
                # This is done so adjacency matrix is symmetric.
                v1, v2 = edges_list_starts[-1], edges_list_ends[-1]
                edges_list_starts.append(v2)
                edges_list_ends.append(v1)
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

    def _random_walks_classic(
        self, width: int, length: int, start_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Allocate memory.
        x_shape = (width * length, self.encoded_state_size)
        x = torch.zeros(x_shape, device=self.device, dtype=torch.int64)
        y = torch.zeros(width * length, device=self.device, dtype=torch.int32)

        # First state in each walk is the start state.
        x[:width, :] = start_state.reshape((-1,))
        y[:width] = 0

        # Main loop.
        for i_step in range(1, length):
            y[i_step * width : (i_step + 1) * width] = i_step
            gen_idx = torch.randint(0, self.definition.n_generators, (width,), device=self.device)
            src = x[(i_step - 1) * width : i_step * width, :]
            dst = x[i_step * width : (i_step + 1) * width, :]
            for j in range(self.definition.n_generators):
                # Go to next state for walks where we chose to use j-th generator on this step.
                mask = gen_idx == j
                prev_states = src[mask, :]
                next_states = torch.zeros_like(prev_states)
                self._apply_generator_batched(j, prev_states, next_states)
                dst[mask, :] = next_states

        return self.decode_states(x), y

    def _random_walks_bfs(
        self, width: int, length: int, start_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_hashes = TorchHashSet()
        x_hashes.add_sorted_hashes(self.hasher.make_hashes(start_state))
        x = [start_state]
        y = [torch.full((1,), 0, device=self.device, dtype=torch.int32)]

        for i_step in range(1, length):
            next_states = self.get_neighbors(x[-1])
            next_states, next_states_hashes, _ = self.get_unique_states(next_states)
            mask = x_hashes.get_mask_to_remove_seen_hashes(next_states_hashes)
            next_states, next_states_hashes = next_states[mask], next_states_hashes[mask]
            layer_size = len(next_states)
            if layer_size == 0:
                break
            if layer_size > width:
                random_indices = torch.randperm(layer_size)[:width]
                layer_size = width
                next_states = next_states[random_indices]
                next_states_hashes = next_states_hashes[random_indices]
            x.append(next_states)
            x_hashes.add_sorted_hashes(next_states_hashes)
            y.append(torch.full((layer_size,), i_step, device=self.device, dtype=torch.int32))
        return self.decode_states(torch.vstack(x)), torch.hstack(y)

    def random_walks(
        self,
        *,
        width=5,
        length=10,
        mode="classic",
        start_state: Union[None, torch.Tensor, np.ndarray, list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random walks on this graph.

        The following modes of random walk generation are supported:

          * "classic" - random walk is a path in this graph starting from `start_state`, where on each step the next
            edge is chosen randomly with equal probability. We generate `width` such random walks independently.
            The output will have exactly ``width*length`` states.
            i-th random walk can be extracted as: ``[x[i+j*width] for j in range(length)]``.
            ``y[i]`` is equal to number of random steps it took to get to state ``x[i]``.
            Note that in this mode a lot of states will have overestimated distance (meaning ``y[i]`` may be larger than
            the length of the shortest path from ``x[i]`` to `start_state`).
            The same state may appear multiple times with different distance in ``y``.
          * "bfs" - we perform Breadth First Search starting from ``start_state`` with one modification: if size of
            next layer is larger than ``width``, only ``width`` states (chosen randomly) will be kept.
            We also remove states from current layer if they appeared on some previous layer (so this also can be
            called "non-backtracking random walk").
            All states in the output are unique. ``y`` still can be overestimated, but it will be closer to the true
            distance than in "classic" mode. Size of output is ``<= width*length``.
            If ``width`` and ``length`` are large enough (``width`` at least as large as largest BFS layer, and
            ``length >= diameter``), this will return all states and true distances to the start state.

        :param width: Number of random walks to generate.
        :param length: Length of each random walk.
        :param start_state: State from which to start random walk. Defaults to the central state.
        :param mode: Type of random walk (see above). Defaults to "classic".
        :return: Pair of tensors ``x, y``. ``x`` contains states. ``y[i]`` is the estimated distance from start state
          to state ``x[i]``.
        """
        start_state = self.encode_states(start_state or self.central_state)
        if mode == "classic":
            return self._random_walks_classic(width, length, start_state)
        elif mode == "bfs":
            return self._random_walks_bfs(width, length, start_state)
        else:
            raise ValueError("Unknown mode:", mode)

    def beam_search(
        self,
        *,
        start_state: Union[torch.Tensor, np.ndarray, list],
        predictor: Predictor,
        beam_width=1000,
        max_iterations=1000,
        return_path=False,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to central state using Beam Search algorithm.

        :param start_state: State from which to star search.
        :param predictor: A heuristic that estimates scores for states (lower score= closer to center).
        :param beam_width: Width of the beam (how many "best" states we consider at each step".
        :param max_iterations: Maximum number of iterations before giving up.
        :param return_path: Whether to return parth (consumes much more memory if True).
        :return: BeamSearchResult containing found path length and 9optionally) the path itself.
        """
        start_states = self.encode_states(start_state)
        layer1, layer1_hashes, _ = self.get_unique_states(start_states)
        all_layers_hashes = [layer1_hashes]

        for i in range(max_iterations):
            if bool(isin_via_searchsorted(self.central_state_hash, layer1_hashes)):
                # Path found.
                path = None
                if return_path:
                    path = self._restore_beam_search_path(all_layers_hashes)
                return BeamSearchResult(True, i, path, self.definition)

            # Create states on the next layer.
            layer2, layer2_hashes, _ = self.get_unique_states(self.get_neighbors(layer1))

            # Pick `beam_width` states with lowest scores.
            if len(layer2) >= beam_width:
                scores = predictor(self.decode_states(layer2))
                idx = torch.argsort(scores)[:beam_width]
                layer2 = layer2[idx, :]
                layer2_hashes = layer2_hashes[idx]
                if self.verbose >= 2:
                    best_score = scores[idx[0]]
                    print(f"Iteration {i}, best score {best_score}.")

            layer1 = layer2
            layer1_hashes = layer2_hashes
            if return_path:
                all_layers_hashes.append(layer1_hashes)

        # Path not found.
        return BeamSearchResult(False, 0, None, self.definition)

    def _restore_beam_search_path(self, hashes: list[torch.Tensor]) -> list[int]:
        """Restores path found by the Beam Search algorithm."""
        inv_graph = CayleyGraph(self.definition.with_inverted_generators())
        assert bool(isin_via_searchsorted(self.central_state_hash, hashes[-1]))
        assert len(hashes[0]) == 1
        path = []  # type: list[int]
        cur_state = self.decode_states(self.encode_states(self.central_state))

        for i in range(len(hashes) - 2, -1, -1):
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
