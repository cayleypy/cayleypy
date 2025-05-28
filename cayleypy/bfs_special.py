"""Special BFS algorithms, optimized for low memory usage."""
import itertools
import math

import numba
import numpy as np

from .cayley_graph import CayleyGraph
from .permutation_utils import inverse_permutation, is_permutation


def bfs_numpy(graph: CayleyGraph, max_diameter: int = 1000000) -> list[int]:
    """Simple version of BFS (from destination_state) using numpy, optimized for memory usage."""
    assert graph.generators_inverse_closed, "Only supports undirected graph."
    assert graph.string_encoder is not None
    assert graph.string_encoder.encoded_length == 1, "Only works on states encoded by single int64."
    perms = [list(x.numpy()) for x in graph.generators]
    perm_funcs = [
        graph.string_encoder.implement_permutation_1d(p) for p in perms]
    pn = len(perms)
    start_state_tensor = graph._encode_states(graph.destination_state).cpu().numpy().reshape(-1)
    start_state = np.array(start_state_tensor, dtype=np.int64)

    # For each generating permutation store which one is its inverse.
    inv_perm_idx = []
    for i in range(pn):
        inv = [j for j in range(pn) if inverse_permutation(
            perms[i]) == perms[j]]
        assert len(inv) == 1
        inv_perm_idx.append(inv[0])

    def _make_states_unique(layer):
        for i1 in range(pn):
            for i2 in range(i1 + 1, pn):
                layer[i1] = np.setdiff1d(
                    layer[i1], layer[i2], assume_unique=True)

    layer0 = [start_state] * pn
    layer1 = [perm_funcs[i](start_state) for i in range(pn)]
    layer1 = [np.setdiff1d(x, start_state, assume_unique=True) for x in layer1]
    _make_states_unique(layer1)
    layer_sizes = [1, len(np.unique(np.hstack(layer1)))]

    for i in range(2, max_diameter + 1):
        layer2 = []
        for i1 in range(pn):
            # All states where we can go from layer1 by permutation i1 (except those that are in layer0).
            next_group = [perm_funcs[i1](layer1[i2])
                          for i2 in range(pn) if i2 != inv_perm_idx[i1]]
            states = np.hstack(next_group)
            states = np.sort(states)
            for i2 in range(pn):
                states = np.setdiff1d(states, layer0[i2], assume_unique=True)
                states = np.setdiff1d(states, layer1[i2], assume_unique=True)
            layer2.append(states)
        _make_states_unique(layer2)
        layer2_size = sum(len(x) for x in layer2)
        if layer2_size == 0:
            break
        layer_sizes.append(layer2_size)
        if graph.verbose >= 2:
            print(f"Layer {i}: {layer2_size} states.")
        layer0, layer1 = layer1, layer2
        if layer2_size >= 10 ** 9:
            graph._free_memory()

    return layer_sizes


def bfs_bitmask(graph: CayleyGraph) -> list[int]:
    """Version of BFS storing all vertices explicitly as bitmasks, using 3 bits of memory per state.

    See https://www.kaggle.com/code/fedimser/memory-efficient-bfs-on-caley-graphs-3bits-per-vx

    :param graph: Cayley graph for which to compute growth function.
    :return: Growth function (layer sizes).
    """
    R = 8  # Chunk prefix size.
    N = len(graph.destination_state)
    assert N > R, f"This algorithm works only for N>{R}."

    graph_size = math.factorial(N)
    chunk_size = math.factorial(R)
    chunks_num = graph_size // chunk_size
    assert chunk_size % 64 == 0
    assert chunks_num * chunk_size == graph_size
    suffix_mask = (2 ** (4 * (N - R)) - 1) << (4 * R)

    # Prepare tables to convert between permutation and its rank.
    model_prefixes = list(itertools.permutations(range(R)))
    THREE = 3
    THREE_MASK = (2 ** THREE) - 1
    prefix_map1 = np.zeros((chunk_size,), dtype=np.int32)  # Maps prefix id to encoded permutation.
    prefix_map2 = np.zeros((2 ** (THREE * R)), dtype=np.int32)  # Maps encoded permutation to prefix id.
    assert len(model_prefixes) == chunk_size
    for i, prefix in enumerate(model_prefixes):
        encoded_prefix = sum(prefix[i] << (THREE * i) for i in range(R))
        prefix_map1[i] = encoded_prefix
        prefix_map2[encoded_prefix] = i

    # Prepare functions to compute permutations.
    assert is_permutation(graph.destination_state), "This version of BFS works only for permutations."
    perms = graph.generators
    enc = graph.string_encoder
    assert enc is not None
    perm_funcs = [enc.implement_permutation_1d(p) for p in perms]
    perm_funcs = [numba.njit("i8[:](i8[:])")(f) for f in perm_funcs]

    def encode_perm(p):
        return sum(p[i] << (4 * i) for i in range(N))

    if graph.verbose >= 2:
        estimated_memory_gb = (math.factorial(N) * 3 / 8) / (2 ** 30)
        print(f"Estimated memory usage: {estimated_memory_gb:.02f}GB.")

    # Credit: https://nimrod.blog/posts/algorithms-behind-popcount/
    @numba.njit("i8(u8[:])")
    def _bit_count(x):
        ans = 0
        for i in range(len(x)):
            n = x[i]
            n = n - ((n >> 1) & 0x5555555555555555)
            n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
            n = (n + (n >> 4)) & 0xF0F0F0F0F0F0F0F
            ans += (n * 0x101010101010101) >> 56
        return ans

    # Ignores suffix.
    @numba.njit("i8(i8,i8[:])", inline="always")
    def permutation_to_rank(p, chunk_map2):
        encoded_prefix = 0
        for i in range(R):
            encoded_prefix |= chunk_map2[(p >> (4 * i)) & 15] << (THREE * i)
        return prefix_map2[encoded_prefix]

    @numba.njit("i8(i8,i8[:])", inline="always")
    def rank_to_permutation(rank, chunk_map1):
        encoded_prefix = prefix_map1[rank]
        ans = 0
        for i in range(R):
            ans |= chunk_map1[(encoded_prefix >> (THREE * i)) & THREE_MASK] << (4 * i)
        return ans

    @numba.njit("(i8[:],u8[:],i8[:])")
    def _materialize_permutations(ans, black, map1):
        ctr = 0
        for i1 in range(chunk_size // 64):
            if black[i1] == 0:
                continue
            mask = black[i1]
            for i2 in range(64):
                if ((mask >> i2) & 1) == 1:
                    rank = 64 * i1 + i2
                    ans[ctr] |= rank_to_permutation(rank, map1)
                    ctr += 1
        assert ctr == len(ans)

    @numba.njit("u8(u8)", inline="always")
    def pow2(x):
        return 1 << x

    @numba.njit("(i8[:],u8[:],i8[:])", inline="always")
    def _paint_gray(perms, gray, map2):
        for i in range(len(perms)):
            perm = perms[i]
            rank = permutation_to_rank(perms[i], map2)
            gray[rank // 64] |= pow2(rank % 64)

    # All possible permutations sharing common suffix of length N-R.
    # We do not store them explicitly, but materialize each time.
    class VertexChunk:
        def __init__(self, suffix):
            self.black = np.zeros((chunk_size // 64,), dtype=np.uint64)
            self.last_layer = np.zeros((chunk_size // 64,), dtype=np.uint64)
            self.gray = np.zeros((chunk_size // 64,), dtype=np.uint64)
            self.changed_on_last_step = False
            self.last_layer_count = 0
            self.encoded_suffix = sum(suffix[i - R] << (4 * i) for i in range(R, N))

            assert len(suffix) == N - R
            # Map indexes in (0..R-1) to actual indexes in prefix, encoded by this permutation.
            self.map1 = np.array([i for i in range(N) if i not in suffix], dtype=np.int64)
            assert len(self.map1) == R
            self.map2 = np.zeros((N,), dtype=np.int64)  # Inverse of map1.
            for i in range(R):
                self.map2[self.map1[i]] = i

        # Returns array of length black_count - explict permutations for black vertices this chunk.
        def materialize_last_layer_permutations(self):
            assert self.changed_on_last_step
            assert self.last_layer_count > 0
            ans = np.full((self.last_layer_count,), self.encoded_suffix, dtype=np.int64)
            _materialize_permutations(ans, self.last_layer, self.map1)
            return ans

        # Paints vertices of given ranks gray.
        # Vertices outside of this chunk are ignored.
        def paint_gray(self, perms):
            _paint_gray(perms, self.gray, self.map2)

        # black |= gray. Clears gray.
        def flush_gray_to_black(self):
            self.gray &= ~self.black  # Now gray contains NEW vertices added on this step.
            self.last_layer_count = _bit_count(self.gray)
            if self.last_layer_count == 0:
                self.changed_on_last_step = False
                self.last_layer[:] = 0
            else:
                self.changed_on_last_step = True
                self.black |= self.gray
                self.last_layer, self.gray = self.gray, self.last_layer
                self.gray[:] = 0

    class CaleyGraphChunkedBfs:
        def __init__(self):
            self.chunks = [VertexChunk(prefix) for prefix in itertools.permutations(range(N), r=N - R)]
            self.chunk_map = {c.encoded_suffix: c for c in self.chunks}
            assert len(self.chunks) == chunks_num

        def paint_gray(self, perms):
            if len(perms) == 1:
                self.chunk_map[perms[0] & suffix_mask].paint_gray(perms)
                return
            perms = np.unique(perms)
            keys = perms & suffix_mask
            group_starts = np.where(np.roll(keys, 1) != keys)[0]
            for i in range(len(group_starts) - 1):
                i1, i2 = group_starts[i], group_starts[i + 1]
                self.chunk_map[keys[i1]].paint_gray(perms[i1:i2])
            i1 = group_starts[-1]
            self.chunk_map[keys[i1]].paint_gray(perms[i1:])

        def flush_gray_to_black(self):
            for c in self.chunks:
                c.flush_gray_to_black()

        def count_last_layer(self):
            return sum([c.last_layer_count for c in self.chunks])

        def bfs(self, max_diameter=100):
            initial_states = np.array([encode_perm(graph.destination_state)], dtype=np.int64)
            self.paint_gray(initial_states)
            self.flush_gray_to_black()
            layer_sizes = [self.count_last_layer()]

            for i in range(1, max_diameter + 1):
                chunks_used = 0
                for c1 in self.chunks:
                    if not c1.changed_on_last_step:
                        continue
                    perms = c1.materialize_last_layer_permutations()
                    neighbors = np.hstack([p(perms) for p in perm_funcs])
                    self.paint_gray(neighbors)
                    chunks_used += 1
                if chunks_used == 0:
                    break
                self.flush_gray_to_black()

                layer_size = self.count_last_layer()
                if layer_size == 0:
                    break
                layer_sizes.append(layer_size)
                if graph.verbose >= 2:
                    print(f"Layer {i} - size {layer_size}.")
            return layer_sizes

    return CaleyGraphChunkedBfs().bfs()
