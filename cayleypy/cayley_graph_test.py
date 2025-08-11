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
    graph = CayleyGraph(graph_def, batch_size=10000)
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
