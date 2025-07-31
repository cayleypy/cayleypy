"""Special BFS algorithm using bitmask, optimized for memory usage."""

import itertools
import math

import numba
import numpy as np

from ..cayley_graph import CayleyGraph
from ..permutation_utils import is_permutation

R = 8  # Chunk prefix size.
CHUNK_SIZE = math.factorial(R)
PREFIX_MAP_1 = np.zeros((CHUNK_SIZE,), dtype=np.int32)  # Maps prefix id to encoded permutation.
PREFIX_MAP_2 = np.zeros((2 ** (3 * R)), dtype=np.int32)  # Maps encoded permutation to prefix id.


def _prepare_prefix_maps():
    """Prepare tables to convert between permutation and its rank."""
    model_prefixes = list(itertools.permutations(range(R)))
    assert len(model_prefixes) == CHUNK_SIZE
    for i, prefix in enumerate(model_prefixes):
        encoded_prefix = sum(prefix[i] << (3 * i) for i in range(R))
        PREFIX_MAP_1[i] = encoded_prefix
        PREFIX_MAP_2[encoded_prefix] = i


_prepare_prefix_maps()


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


@numba.njit("i8(i8,i8[:])", inline="always")
def permutation_to_rank(p, chunk_map2):
    # Ignores suffix.
    encoded_prefix = 0
    for i in range(R):
        encoded_prefix |= chunk_map2[(p >> (4 * i)) & 15] << (3 * i)
    return PREFIX_MAP_2[encoded_prefix]


@numba.njit("i8(i8,i8[:])", inline="always")
def rank_to_permutation(rank, chunk_map1):
    encoded_prefix = PREFIX_MAP_1[rank]
    ans = 0
    for i in range(R):
        ans |= chunk_map1[(encoded_prefix >> (3 * i)) & 7] << (4 * i)
    return ans


@numba.njit("(i8[:],u8[:],i8[:])")
def _materialize_permutations(ans, black, map1):
    ctr = 0
    for i1 in range(CHUNK_SIZE // 64):
        if black[i1] == 0:
            continue
        mask = black[i1]
        for i2 in range(64):
            if ((mask >> i2) & 1) == 1:
                rank = 64 * i1 + i2
                ans[ctr] |= rank_to_permutation(rank, map1)
                ctr += 1
    assert ctr == len(ans)


@numba.njit("(i8[:],u8[:],i8[:])")
def _paint_gray(perms, gray, map2):
    for i in range(len(perms)):
        rank = permutation_to_rank(perms[i], map2)
        gray[rank // 64] |= 1 << (rank % 64)


def _encode_perm(p):
    return sum(p[i] << (4 * i) for i in range(len(p)))


class VertexChunk:
    """All possible permutations sharing common suffix of length N-R.

    We do not store them explicitly, but materialize each time.
    """

    def __init__(self, n, suffix):
        self.black = np.zeros((CHUNK_SIZE // 64,), dtype=np.uint64)
        self.last_layer = np.zeros((CHUNK_SIZE // 64,), dtype=np.uint64)
        self.gray = np.zeros((CHUNK_SIZE // 64,), dtype=np.uint64)
        self.changed_on_last_step = False
        self.last_layer_count = 0
        self.encoded_suffix = sum(suffix[i - R] << (4 * i) for i in range(R, n))

        assert len(suffix) == n - R
        # Map indexes in (0..R-1) to actual indexes in prefix, encoded by this permutation.
        self.map1 = np.array([i for i in range(n) if i not in suffix], dtype=np.int64)
        assert len(self.map1) == R
        self.map2 = np.zeros((n,), dtype=np.int64)  # Inverse of map1.
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


class CayleyGraphChunkedBfs:
    """Class to run the special BFS algorithm."""

    def __init__(self, graph: CayleyGraph):
        n = graph.definition.state_size
        self.graph = graph
        self.chunks = [VertexChunk(n, prefix) for prefix in itertools.permutations(range(n), r=n - R)]
        self.chunk_map = {c.encoded_suffix: c for c in self.chunks}

        graph_size = math.factorial(n)
        chunks_num = graph_size // CHUNK_SIZE
        assert CHUNK_SIZE % 64 == 0
        assert chunks_num * CHUNK_SIZE == graph_size
        assert len(self.chunks) == chunks_num
        self.suffix_mask = (2 ** (4 * (n - R)) - 1) << (4 * R)

        # Prepare functions to compute permutations.
        assert is_permutation(graph.definition.central_state), "This version of BFS works only for permutations."
        perms = graph.definition.generators_permutations
        enc = graph.string_encoder
        assert enc is not None
        perm_funcs = [enc.implement_permutation_1d(p) for p in perms]
        self.perm_funcs = [numba.njit("i8[:](i8[:])")(f) for f in perm_funcs]

    def paint_gray(self, perms):
        if len(perms) == 1:
            self.chunk_map[perms[0] & self.suffix_mask].paint_gray(perms)
            return
        perms = np.unique(perms)
        keys = perms & self.suffix_mask
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
        return sum(c.last_layer_count for c in self.chunks)

    def bfs(self, max_diameter=10**6):
        initial_states = np.array([_encode_perm(self.graph.definition.central_state)], dtype=np.int64)
        self.paint_gray(initial_states)
        self.flush_gray_to_black()
        layer_sizes = [self.count_last_layer()]

        for i in range(1, max_diameter + 1):
            chunks_used = 0
            for c1 in self.chunks:
                if not c1.changed_on_last_step:
                    continue
                perms = c1.materialize_last_layer_permutations()
                neighbors = np.hstack([p(perms) for p in self.perm_funcs])
                self.paint_gray(neighbors)
                chunks_used += 1
            if chunks_used == 0:
                break
            self.flush_gray_to_black()

            layer_size = self.count_last_layer()
            if layer_size == 0:
                break
            layer_sizes.append(layer_size)
            if self.graph.verbose >= 2:
                print(f"Layer {i} - size {layer_size}.")
        return layer_sizes


def bfs_bitmask(graph: CayleyGraph, max_diameter: int = 10**6) -> list[int]:
    """Version of BFS storing all vertices explicitly as bitmasks, using 3 bits of memory per state.

    See https://www.kaggle.com/code/fedimser/memory-efficient-bfs-on-caley-graphs-3bits-per-vx

    :param graph: Cayley graph for which to compute growth function.
    :param max_diameter:  maximal number of BFS iterations.
    :return: Growth function (layer sizes).
    """
    assert graph.definition.is_permutation_group(), "Only works for permutations."
    n = graph.definition.state_size
    assert n > R, f"This algorithm works only for N>{R}."
    if graph.verbose >= 2:
        estimated_memory_gb = (math.factorial(n) * 3 / 8) / (2**30)
        print(f"Estimated memory usage: {estimated_memory_gb:.02f}GB.")
    return CayleyGraphChunkedBfs(graph).bfs(max_diameter=max_diameter)
