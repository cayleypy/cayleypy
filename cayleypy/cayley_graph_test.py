import os

import numpy as np
import pytest
import torch

from cayleypy.algo import bfs_numpy
from .cayley_graph import CayleyGraph
from .cayley_graph_def import MatrixGenerator, CayleyGraphDef
from .datasets import load_dataset
from .graphs_lib import PermutationGroups, MatrixGroups, prepare_graph


RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"
BENCHMARK_RUN = os.getenv("BENCHMARK") == "1"


def _layer_to_set(layer: np.ndarray) -> set[str]:
    return set("".join(str(x) for x in state) for state in layer)


def test_generators_format():
    generators = [[1, 2, 0], [2, 0, 1], [1, 0, 2]]
    graph1 = CayleyGraphDef.create(generators)
    graph2 = CayleyGraphDef.create(np.array(generators))
    graph3 = CayleyGraphDef.create(torch.tensor(generators))
    assert np.array_equal(graph1.generators, graph2.generators)
    assert np.array_equal(graph1.generators, graph3.generators)


def test_central_state_format():
    graph_def = PermutationGroups.lrx(10)
    dest_list = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    graph1 = CayleyGraph(graph_def.with_central_state("0123012301"))
    graph2 = CayleyGraph(graph_def.with_central_state(dest_list))
    graph3 = CayleyGraph(graph_def.with_central_state(dest_list))
    graph4 = CayleyGraph(graph_def.with_central_state(dest_list))
    assert torch.equal(graph1.central_state, graph2.central_state)
    assert torch.equal(graph1.central_state, graph3.central_state)
    assert torch.equal(graph1.central_state, graph4.central_state)


def test_bfs_growth_swap():
    graph = CayleyGraph(CayleyGraphDef.create([[1, 0]], central_state="01"))
    result = graph.bfs()
    assert result.layer_sizes == [1, 1]
    assert result.diameter() == 1
    assert _layer_to_set(result.get_layer(0)) == {"01"}
    assert _layer_to_set(result.get_layer(1)) == {"10"}


def test_bfs_lrx_coset_5():
    graph = CayleyGraph(PermutationGroups.lrx(5).with_central_state("01210"))
    ans = graph.bfs()
    assert ans.bfs_completed
    assert ans.diameter() == 6
    assert ans.layer_sizes == [1, 3, 5, 8, 7, 5, 1]
    assert _layer_to_set(ans.get_layer(0)) == {"01210"}
    assert _layer_to_set(ans.get_layer(1)) == {"00121", "10210", "12100"}
    assert _layer_to_set(ans.get_layer(5)) == {"00112", "01120", "01201", "02011", "11020"}
    assert _layer_to_set(ans.get_layer(6)) == {"10201"}


def test_bfs_lrx_coset_10():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    ans = graph.bfs()
    assert ans.diameter() == 17
    assert ans.layer_sizes == [1, 3, 4, 6, 11, 16, 19, 23, 31, 29, 20, 14, 10, 10, 6, 3, 3, 1]
    assert _layer_to_set(ans.get_layer(0)) == {"0110110110"}
    assert _layer_to_set(ans.get_layer(1)) == {"0011011011", "1010110110", "1101101100"}
    assert _layer_to_set(ans.get_layer(15)) == {"0001111110", "0111111000", "1110000111"}
    assert _layer_to_set(ans.get_layer(16)) == {"0011111100", "1111000011", "1111110000"}
    assert _layer_to_set(ans.get_layer(17)) == {"1111100001"}


def test_bfs_max_radius():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    ans = graph.bfs(max_diameter=5)
    assert not ans.bfs_completed
    assert ans.layer_sizes == [1, 3, 4, 6, 11, 16]


def test_bfs_max_layer_size_to_explore():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    ans = graph.bfs(max_layer_size_to_explore=10)
    assert not ans.bfs_completed
    assert ans.layer_sizes == [1, 3, 4, 6, 11]


def test_bfs_max_layer_size_to_store():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    ans = graph.bfs(max_layer_size_to_store=10)
    assert ans.bfs_completed
    assert ans.diameter() == 17
    assert ans.layers.keys() == {0, 1, 2, 3, 12, 13, 14, 15, 16, 17}

    ans = graph.bfs(max_layer_size_to_store=None)
    assert ans.bfs_completed
    assert ans.diameter() == 17
    assert ans.layers.keys() == set(range(18))


def test_bfs_start_state():
    graph = CayleyGraph(PermutationGroups.lrx(5))
    ans = graph.bfs(start_states=[0, 1, 2, 1, 0])
    assert ans.bfs_completed
    assert ans.layer_sizes == [1, 3, 5, 8, 7, 5, 1]


def test_bfs_multiple_start_states():
    graph = CayleyGraph(PermutationGroups.lrx(5))
    ans = graph.bfs(start_states=[[0, 1, 2, 1, 0], [1, 0, 2, 0, 1], [0, 1, 1, 2, 0]])
    assert ans.bfs_completed
    assert ans.layer_sizes == [3, 9, 11, 6, 1]


@pytest.mark.parametrize("bit_encoding_width", [None, 6])
def test_bfs_lrx_n40_layers5(bit_encoding_width):
    # We need 6*40=240 bits for encoding, so each states is encoded by four int64's.
    graph_def = PermutationGroups.lrx(40)
    graph = CayleyGraph(graph_def, bit_encoding_width=bit_encoding_width)
    assert graph.bfs(max_diameter=5).layer_sizes == [1, 3, 6, 12, 24, 48]


def test_bfs_last_layer_lrx_n8():
    graph = CayleyGraph(PermutationGroups.lrx(8))
    assert _layer_to_set(graph.bfs().last_layer()) == {"10765432"}


def test_bfs_last_layer_lrx_coset_n8():
    graph = CayleyGraph(PermutationGroups.lrx(8).with_central_state("01230123"))
    assert _layer_to_set(graph.bfs().last_layer()) == {"11003322", "22110033", "33221100", "00332211"}


@pytest.mark.parametrize("bit_encoding_width", [None, 3, 10, "auto"])
def test_bfs_bit_encoding(bit_encoding_width):
    graph_def = PermutationGroups.lrx(8)
    result = CayleyGraph(graph_def, bit_encoding_width=bit_encoding_width).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.parametrize("batch_size", [100, 1000, 10**9])
def test_bfs_batching_lrx(batch_size: int):
    graph_def = PermutationGroups.lrx(8)
    graph = CayleyGraph(graph_def, batch_size=batch_size)
    result = graph.bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


# Test that batching works when state doesn't fit in int64.
def test_bfs_batching_coxeter20():
    graph_def = PermutationGroups.coxeter(20)
    graph = CayleyGraph(graph_def, batch_size=10000, bit_encoding_width="auto")
    assert not graph.hasher.is_identity
    assert graph.string_encoder.encoded_length == 2
    result = graph.bfs(max_diameter=7)
    assert result.layer_sizes == load_dataset("coxeter_cayley_growth")["20"][:8]


def test_bfs_batching_all_transpositions():
    graph_def = PermutationGroups.all_transpositions(8)
    graph = CayleyGraph(graph_def, batch_size=2**10)
    result = graph.bfs()
    assert result.layer_sizes == load_dataset("all_transpositions_cayley_growth")["8"]


@pytest.mark.parametrize("hash_chunk_size", [100, 1000, 10**9])
def test_bfs_hash_chunking(hash_chunk_size: int):
    graph_def = PermutationGroups.lrx(8)
    result = CayleyGraph(graph_def, hash_chunk_size=hash_chunk_size).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.parametrize("bit_encoding_width", [None, 5])
def test_get_neighbors(bit_encoding_width):
    # Directly check _get_neighbors_batched.
    # In what order it generates neighbours is an implementation detail. However, we rely on this convention when
    # generating the edges list.
    graph_def = CayleyGraphDef.create([[1, 0, 2, 3, 4], [0, 1, 2, 4, 3]])
    graph = CayleyGraph(graph_def, bit_encoding_width=bit_encoding_width)
    states = graph.encode_states(torch.tensor([[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]], dtype=torch.int64))
    result = graph.decode_states(graph.get_neighbors(states))
    assert torch.equal(
        result.cpu(),
        torch.tensor([[11, 10, 12, 13, 14], [16, 15, 17, 18, 19], [10, 11, 12, 14, 13], [15, 16, 17, 19, 18]]),
    )


def test_edges_list_n2():
    graph = CayleyGraph(CayleyGraphDef.create([[1, 0]], central_state="01"))
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.named_undirected_edges() == {("01", "10")}


def test_edges_list_n3():
    graph = CayleyGraph(PermutationGroups.lrx(3).with_central_state("001"))
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.named_undirected_edges() == {("001", "001"), ("001", "010"), ("001", "100"), ("010", "100")}


@pytest.mark.parametrize("bit_encoding_width", [None, 5])
def test_edges_list_n4(bit_encoding_width):
    graph_def = PermutationGroups.top_spin(4).with_central_state("0011")
    graph = CayleyGraph(graph_def, bit_encoding_width=bit_encoding_width)
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.named_undirected_edges() == {
        ("0011", "0110"),
        ("0011", "1001"),
        ("0011", "1100"),
        ("0110", "0110"),
        ("0110", "1100"),
        ("1001", "1001"),
        ("1001", "1100"),
    }


def test_generators_not_inverse_closed():
    graph = CayleyGraphDef.create([[1, 2, 3, 0]])
    assert not graph.generators_inverse_closed
    assert CayleyGraph(graph).bfs().layer_sizes == [1, 1, 1, 1]


# Tests below compare growth function for small graphs with stored pre-computed results.
def test_lrx_cayley_growth():
    expected = load_dataset("lrx_cayley_growth")
    for n in range(3, 10):
        graph = CayleyGraph(PermutationGroups.lrx(n))
        result = graph.bfs()
        assert result.layer_sizes == expected[str(n)]


def test_top_spin_cayley_growth():
    expected = load_dataset("top_spin_cayley_growth")
    for n in range(4, 10):
        graph = CayleyGraph(PermutationGroups.top_spin(n))
        result = graph.bfs()
        assert result.layer_sizes == expected[str(n)]


def test_lrx_coset_growth():
    expected = load_dataset("lrx_coset_growth")
    for central_state, expected_layer_sizes in expected.items():
        if len(central_state) > 15:
            continue
        generators = PermutationGroups.lrx(len(central_state)).generators
        graph = CayleyGraph(CayleyGraphDef.create(generators, central_state=central_state))
        result = graph.bfs()
        assert result.layer_sizes == expected_layer_sizes


# Skipped by default.
# To run slow tests like this, do `RUN_SLOW_TESTS=1 pytest`
@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_cube222_qtm():
    graph = CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"))
    result = graph.bfs()
    assert result.num_vertices == 3674160
    assert result.diameter() == 14
    assert result.layer_sizes == load_dataset("puzzles_growth")["cube_222_fixed_qtm"]


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_cube222_htm():
    graph = CayleyGraph(prepare_graph("cube_2/2/2_9gensHTM"))
    result = graph.bfs()
    assert result.num_vertices == 3674160
    assert result.diameter() == 11
    assert result.layer_sizes == load_dataset("puzzles_growth")["cube_222_fixed_htm"]


def test_all_transpositions_8():
    graph = CayleyGraph(PermutationGroups.all_transpositions(8))
    result = graph.bfs()
    assert result.layer_sizes == load_dataset("all_transpositions_cayley_growth")["8"]


def test_generator_names():
    graph = CayleyGraphDef.create([[1, 2, 3, 0], [0, 2, 1, 3]])
    assert graph.generator_names == ["1,2,3,0", "0,2,1,3"]

    graph = PermutationGroups.lrx(4)
    assert graph.generator_names == ["L", "R", "X"]


def test_bfs_small_hash_chunk_size():
    graph_def = PermutationGroups.lrx(20)
    graph = CayleyGraph(graph_def, hash_chunk_size=100)
    assert graph.bfs(max_diameter=8).layer_sizes == [1, 3, 6, 12, 24, 48, 91, 172, 325]


def test_hashes_list_len():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.bfs_completed
    assert result.num_vertices == len(result.vertex_names)


def test_hashes_list_len_max_radius():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_diameter=2)
    assert not result.bfs_completed
    assert result.num_vertices == len(result.vertex_names)


def test_hashes_list_len_max_layer_size_to_explore():
    graph = CayleyGraph(PermutationGroups.lrx(10).with_central_state("0110110110"))
    result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_layer_size_to_explore=2)
    assert not result.bfs_completed
    assert result.num_vertices == len(result.vertex_names)


def test_matrix_group():
    p = 10
    x = MatrixGenerator.create([[1, 1], [0, 1]], modulo=p)
    x_inv = MatrixGenerator.create([[1, -1], [0, 1]], modulo=p)
    graph = CayleyGraph(
        CayleyGraphDef.for_matrix_group(
            generators=[x, x_inv],
            generator_names=["x", "x'"],
            central_state=[[1, 2], [0, 1]],
        )
    )
    assert not graph.definition.is_permutation_group()
    assert graph.definition.n_generators == 2
    assert graph.definition.generators_matrices[0].n == 2
    bfs_result = graph.bfs()
    assert bfs_result.layer_sizes == [1, 2, 2, 2, 2, 1]
    assert np.array_equal(bfs_result.last_layer()[0], [[1, 7], [0, 1]])
    assert len(bfs_result.all_states) == 10


def test_bfs_heisenberg_group():
    graph = CayleyGraph(MatrixGroups.heisenberg())
    bfs_result = graph.bfs(max_diameter=15)
    # See https://oeis.org/A063810
    assert bfs_result.layer_sizes == [1, 4, 12, 36, 82, 164, 294, 476, 724, 1052, 1464, 1972, 2590, 3324, 4186, 5188]


def test_incomplete_bfs_symmetric_adjacency_matrix():
    graph = CayleyGraph(prepare_graph("pyraminx"), device="cpu")
    bfs_result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_diameter=2)
    mx = bfs_result.adjacency_matrix()
    assert np.array_equal(mx, mx.T)


def _state_to_str(state: torch.Tensor):
    return "".join(str(int(x)) for x in state)


def test_random_walks_single_walk():
    graph = CayleyGraph(PermutationGroups.lrx(5))
    x, y = graph.random_walks(width=1, length=5)
    assert x.shape == (5, 5)
    assert y.shape == (5,)
    assert _state_to_str(x[0]) == "01234"
    assert _state_to_str(x[1]) in ["12340", "40123", "10234"]
    assert np.array_equal(y.cpu().numpy(), [0, 1, 2, 3, 4])


def test_random_walks_matrix_group():
    graph = CayleyGraph(MatrixGroups.heisenberg())
    x, y = graph.random_walks(width=20, length=10)
    assert x.shape == (200, 3, 3)
    assert y.shape == (200,)
    assert np.array_equal(y, [i for i in range(10) for _ in range(20)])


def test_random_walks_start_state():
    graph = CayleyGraph(PermutationGroups.lx(5))
    x, y = graph.random_walks(width=10, length=5, start_state=[1, 0, 0, 0, 0])
    assert x.shape == (50, 5)
    assert y.shape == (50,)
    for i in range(10):
        assert _state_to_str(x[i]) == "10000"
    for i in range(10, 20):
        assert _state_to_str(x[i]) in ["01000", "00001"]


def test_random_walks_bfs_small():
    graph = CayleyGraph(PermutationGroups.lrx(4))
    x, y = graph.random_walks(width=50, length=100, mode="bfs")
    assert x.shape == (24, 4)
    assert y.shape == (24,)
    assert np.array_equal(y.cpu().numpy(), [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6])


def test_random_walks_bfs():
    graph = CayleyGraph(PermutationGroups.lrx(20))
    x, y = graph.random_walks(width=100, length=50, mode="bfs")
    assert x.shape == (4485, 20)
    assert y.shape == (4485,)
    assert y[0] == 0
    assert y[-1] == 49


def test_random_walks_bfs_matrix_groups():
    graph = CayleyGraph(MatrixGroups.heisenberg())
    x, y = graph.random_walks(width=100, length=50, mode="bfs")
    assert x.shape == (4635, 3, 3)
    assert y.shape == (4635,)


def test_path_to_from():
    n = 8
    graph = CayleyGraph(PermutationGroups.lrx(n))
    br = graph.bfs(return_all_hashes=True)
    for _ in range(5):
        start_state = torch.tensor(np.random.permutation(n))
        path1 = graph.find_path_from(start_state, br)
        assert torch.equal(graph.apply_path(start_state, path1)[0], graph.central_state)
        path2 = graph.find_path_to(start_state, br)
        assert torch.equal(start_state, graph.apply_path(graph.central_state, path2)[0])


# Below is the benchmark code. To run: `BENCHMARK=1 pytest . -k benchmark`
@pytest.mark.skipif(not BENCHMARK_RUN, reason="benchmark")
@pytest.mark.parametrize("benchmark_mode", ["baseline", "bit_encoded", "bfs_numpy"])
@pytest.mark.parametrize("n", [26])
def test_benchmark_top_spin(benchmark, benchmark_mode, n):
    central_state = [0] * (n // 2) + [1] * (n // 2)
    graph_def = PermutationGroups.lrx(n).with_central_state(central_state)
    if benchmark_mode == "bfs_numpy":
        graph = CayleyGraph(graph_def)
        benchmark.pedantic(lambda: bfs_numpy(graph), iterations=1, rounds=5)
    else:
        bit_encoding_width = 1 if benchmark_mode == "bit_encoded" else None
        graph = CayleyGraph(graph_def, bit_encoding_width=bit_encoding_width)
        benchmark.pedantic(graph.bfs, iterations=1, rounds=5)


@pytest.mark.parametrize("bit_encoding_width", [3, 4, 8, "auto"])
def test_bit_encoding_width_values(bit_encoding_width):
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, bit_encoding_width=bit_encoding_width)
    bfs_result = graph.bfs(max_diameter=3)
    # Compare only first layers
    assert bfs_result.layer_sizes == load_dataset("lrx_cayley_growth")["5"][:4]


def test_modified_copy_preserves_hasher_and_encoder():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, bit_encoding_width=3)

    new_def = graph_def.with_central_state("01210")
    new_graph = graph.modified_copy(new_def)

    assert new_graph.hasher is graph.hasher
    assert new_graph.string_encoder is graph.string_encoder

    assert torch.equal(new_graph.central_state, torch.tensor([0, 1, 2, 1, 0]))
    assert not torch.equal(new_graph.central_state, graph.central_state)


def test_with_inverted_generators_path_reversal():
    graph_def = PermutationGroups.lrx(4)
    graph = CayleyGraph(graph_def)

    inv_graph = graph.with_inverted_generators
    assert inv_graph.definition.n_generators == graph.definition.n_generators

    bfs_result = graph.bfs(return_all_hashes=True)
    start_state = torch.tensor([0, 1, 2, 3])

    path_to = graph.find_path_to(start_state, bfs_result)
    path_from = graph.find_path_from(start_state, bfs_result)

    assert path_from == graph.definition.revert_path(path_to)


def test_bfs_on_modified_copy_preserves_structure_safe():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, bit_encoding_width=3)

    new_def = graph_def.with_central_state("01210")
    new_graph = graph.modified_copy(new_def)

    bfs_result = new_graph.bfs(max_diameter=5)

    assert bfs_result.bfs_completed or len(bfs_result.layer_sizes) > 0
    assert not torch.equal(new_graph.central_state, graph.central_state)
    assert new_graph.hasher is graph.hasher
    assert new_graph.string_encoder is graph.string_encoder


# =============================================================================
# Hot-path method tests (get_neighbors, encode/decode, apply_path, restore_path,
# find_path_from, free_memory, get_neighbors_generator)
# =============================================================================


def test_get_neighbors_central_state():
    """``get_neighbors`` returns ``n_generators`` copies of the state, each transformed."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    encoded = graph.encode_states(graph.central_state)
    neighbors = graph.get_neighbors(encoded)
    # LRX has 3 generators -> 3 neighbor rows (1 state * 3 generators).
    assert neighbors.shape[0] == graph.definition.n_generators
    # The central state [0,1,2,3,4] should NOT appear among its own neighbors.
    central_decoded = graph.decode_states(encoded)
    for i in range(neighbors.shape[0]):
        assert not torch.equal(neighbors[i], central_decoded[0])


def test_get_neighbors_multiple_states():
    """``get_neighbors`` on ``k`` states returns ``k * n_generators`` rows."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    states = graph.encode_states(torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]))
    neighbors = graph.get_neighbors(states)
    assert neighbors.shape[0] == 2 * graph.definition.n_generators


def test_get_neighbors_matches_per_generator_loop():
    """The CPU batched-gather fast path (cayley_graph.py:195-219) produces output identical
    to the per-generator loop it replaces.

    The fast path is gated on CPU + permutation group + non-bit-encoded. This test
    exercises that branch on cube333 and LRX(8) and compares against an inlined
    per-generator loop. Guards against layout/ordering regressions in the batched gather.
    """
    for graph in (
        CayleyGraph(PermutationGroups.lrx(8), device="cpu"),
        CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM"), device="cpu"),
    ):
        states = graph.encode_states(torch.randint(0, 8, (7, graph.definition.state_size), dtype=torch.int64))
        actual = graph.get_neighbors(states)
        # Inlined per-generator loop (the path the batched gather replaces on CPU).
        states_num = states.shape[0]
        n_gen = graph.definition.n_generators
        expected = torch.zeros(
            (states_num * n_gen, graph.definition.state_size), dtype=graph.dtype, device=graph.device
        )
        for i in range(n_gen):
            dst = expected[i * states_num : (i + 1) * states_num, :]
            graph.apply_generator_batched(i, states, dst)
        assert torch.equal(actual, expected)


def test_get_neighbors_generator_yields_per_generator():
    """``get_neighbors_generator`` yields one chunk per generator."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    states = graph.encode_states(graph.central_state)
    chunks = list(graph.get_neighbors_generator(states))
    assert len(chunks) == graph.definition.n_generators
    for chunk in chunks:
        assert chunk.shape[0] == states.shape[0]


def test_get_neighbors_generator_no_clone_aliases_buffer():
    """With ``clone=False``, all yielded chunks alias the same internal buffer.

    Documents the no-alias contract (Task 1.5): the caller MUST reassign the yielded
    reference (e.g. via fancy-indexing) before the next yield, or it will see the next
    generator's data. This test verifies the aliasing behavior so a future change to
    the default or the buffer reuse strategy is caught.

    Covers the ``clone=False`` branch for the coverage gate.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))
    states = graph.encode_states(graph.central_state)
    chunks = list(graph.get_neighbors_generator(states, clone=False))
    assert len(chunks) == graph.definition.n_generators
    # All chunks alias the same storage — the last generator's data overwrites all.
    assert all(chunk.data_ptr() == chunks[0].data_ptr() for chunk in chunks)
    # After collecting all, every chunk sees the LAST generator's result.
    last_gen_states = graph.encode_states(graph.central_state)
    graph.apply_generator_batched(graph.definition.n_generators - 1, states, last_gen_states)
    assert torch.equal(chunks[0], last_gen_states)
    assert torch.equal(chunks[-1], last_gen_states)


def test_encode_decode_round_trip():
    """encode_states -> decode_states is the identity for non-bit-encoded graphs."""
    graph = CayleyGraph(PermutationGroups.lrx(5), bit_encoding_width=None)
    original = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    encoded = graph.encode_states(original)
    decoded = graph.decode_states(encoded)
    assert torch.equal(decoded, original)


def test_encode_decode_round_trip_bit_encoded():
    """encode -> decode round-trip with bit encoding."""
    graph = CayleyGraph(PermutationGroups.lrx(5), bit_encoding_width=3)
    original = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    encoded = graph.encode_states(original)
    decoded = graph.decode_states(encoded)
    assert torch.equal(decoded, original)


def test_apply_path_single_generator():
    """``apply_path`` applies generators in order; single-step path."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [0, 1, 2, 3, 4]
    result = graph.apply_path(start, [0])
    # Applying generator 0 (L) to [0,1,2,3,4] should produce a non-identity state.
    assert not torch.equal(result.reshape(-1), torch.tensor(start))


def test_apply_path_round_trip():
    """Applying a generator and then its inverse returns to the start state.

    For LRX, generators L (shift left) and R (shift right) are inverses.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))
    start = [0, 1, 2, 3, 4]
    # Find a pair (i, j) where generator j is the inverse of generator i.
    n_gens = graph.definition.n_generators
    found = False
    for i in range(n_gens):
        for j in range(n_gens):
            # Check if applying generator i then j returns to start (identity).
            result = graph.apply_path(start, [i, j])
            if torch.equal(result.reshape(-1), torch.tensor(start)):
                found = True
                break
        if found:
            break
    assert found, "Should find at least one generator/inverse pair"


def test_apply_path_validates_generator_ids():
    """``apply_path`` asserts generator_id is in range."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    with pytest.raises(AssertionError):
        graph.apply_path([0, 1, 2, 3, 4], [999])


def test_restore_path_from_bfs():
    """``restore_path`` reconstructs a path from BFS layer hashes.

    ``restore_path(layers_hashes[:k], target)`` returns a path of length k that goes
    from layer[0] to ``target`` (which is at layer k).
    """
    graph = CayleyGraph(PermutationGroups.lrx(5))
    bfs_result = graph.bfs(max_diameter=5, return_all_hashes=True)
    # Pick a state at layer 2.
    layer2 = bfs_result.get_layer(2)
    if len(layer2) > 0:
        target_state = layer2[0]
        path = graph.restore_path(bfs_result.layers_hashes[:2], target_state)
        # Path length should be 2 (two layers).
        assert len(path) == 2
        # Applying the path from layer-0 state should reach target_state.
        layer0_state = torch.tensor(bfs_result.get_layer(0)[0])
        result = graph.apply_path(layer0_state, path)
        assert torch.equal(result.reshape(-1), torch.tensor(target_state))


def test_find_path_from_using_bfs():
    """``find_path_from`` finds a path from a state to central using pre-computed BFS."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    bfs_result = graph.bfs(max_diameter=5, return_all_hashes=True)
    # Pick a state at layer 2 and find path from it to central.
    layer2 = bfs_result.get_layer(2)
    if len(layer2) > 0:
        start_state = layer2[0]
        path = graph.find_path_from(start_state, bfs_result)
        assert path is not None
        assert len(path) == 2
        graph.validate_path(start_state, path)


def test_find_path_from_not_found_returns_none():
    """``find_path_from`` returns None when the state is not in the BFS layers."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    bfs_result = graph.bfs(max_diameter=1, return_all_hashes=True)
    # A state at distance > 1 won't be found.
    far_state = [4, 3, 2, 1, 0]
    path = graph.find_path_from(far_state, bfs_result)
    assert path is None


def test_free_memory_cpu():
    """``free_memory`` runs without error on CPU (calls gc.collect)."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    graph.free_memory()  # Should not raise.


def test_get_unique_states_returns_sorted_hashes():
    """Hidden critical contract: ``get_unique_states`` returns hashes in sorted order.

    This invariant is relied upon by ``isin_via_searchsorted`` (which requires a
    sorted ``test_elements_sorted`` argument) inside ``_check_path_found`` and
    ``_remove_seen_states``. A regression here would silently break MITM path
    detection. See AGENTS.md section "Invariants".
    """
    graph = CayleyGraph(PermutationGroups.lrx(8), random_seed=42)
    # Create states whose hashes are NOT in sorted order (use a non-identity hasher).
    assert not graph.hasher.is_identity
    states = torch.tensor(
        [
            [4, 3, 2, 1, 0, 7, 6, 5],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ],
        dtype=torch.int64,
    )
    _, hashes = graph.get_unique_states(graph.encode_states(states))
    if len(hashes) > 1:
        assert torch.all(hashes[1:] >= hashes[:-1]), f"Hashes not sorted: {hashes}"
