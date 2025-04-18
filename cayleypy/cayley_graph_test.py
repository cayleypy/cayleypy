import os

import numpy as np
import pytest
import torch

from cayleypy import CayleyGraph, prepare_graph, load_dataset

FAST_RUN = os.getenv("FAST") == "1"
BENCHMARK_RUN = os.getenv("BENCHMARK") == "1"


def test_generators_format():
    generators = [[1, 2, 0], [2, 0, 1], [1, 0, 2]]
    graph1 = CayleyGraph(generators)
    graph2 = CayleyGraph(np.array(generators))
    graph3 = CayleyGraph(torch.tensor(generators))
    assert torch.equal(graph1.generators, graph2.generators)
    assert torch.equal(graph1.generators, graph3.generators)


def test_destination_format():
    generators = prepare_graph("lrx", n=10)[0]
    dest_list = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    graph1 = CayleyGraph(generators, dest="0123012301")
    graph2 = CayleyGraph(generators, dest=dest_list)
    graph3 = CayleyGraph(generators, dest=np.array(dest_list))
    graph4 = CayleyGraph(generators, dest=torch.tensor(dest_list))
    assert torch.equal(graph1.destination_state, graph2.destination_state)
    assert torch.equal(graph1.destination_state, graph3.destination_state)
    assert torch.equal(graph1.destination_state, graph4.destination_state)


def test_bfs_growth_swap():
    graph = CayleyGraph([[1, 0]], dest="01")
    result = graph.bfs()
    assert result.layer_sizes == [1, 1]
    assert result.diameter() == 1
    assert result.get_layer(0) == ["01"]
    assert result.get_layer(1) == ["10"]


def test_bfs_lrx_coset_5():
    graph = CayleyGraph(prepare_graph("lrx", n=5)[0], dest="01210")
    ans = graph.bfs()
    assert ans.bfs_completed
    assert ans.diameter() == 6
    assert ans.layer_sizes == [1, 3, 5, 8, 7, 5, 1]
    assert ans.get_layer(0) == ["01210"]
    assert set(ans.get_layer(1)) == {"00121", "10210", "12100"}
    assert set(ans.get_layer(5)) == {"00112", "01120", "01201", "02011", "11020"}
    assert ans.get_layer(6) == ["10201"]


def test_bfs_lrx_coset_10():
    graph = CayleyGraph(prepare_graph("lrx", n=10)[0], dest="0110110110")
    ans = graph.bfs()
    assert ans.diameter() == 17
    assert ans.layer_sizes == [1, 3, 4, 6, 11, 16, 19, 23, 31, 29, 20, 14, 10, 10, 6, 3, 3, 1]
    assert ans.get_layer(0) == ["0110110110"]
    assert set(ans.get_layer(1)) == {"0011011011", "1010110110", "1101101100"}
    assert set(ans.get_layer(15)) == {"0001111110", "0111111000", "1110000111"}
    assert set(ans.get_layer(16)) == {"0011111100", "1111000011", "1111110000"}
    assert ans.get_layer(17) == ["1111100001"]


def test_bfs_max_radius():
    graph = CayleyGraph(prepare_graph("lrx", n=10)[0], dest="0110110110")
    ans = graph.bfs(max_diameter=5)
    assert not ans.bfs_completed
    assert ans.layer_sizes == [1, 3, 4, 6, 11, 16]


def test_bfs_max_layer_size_to_explore():
    graph = CayleyGraph(prepare_graph("lrx", n=10)[0], dest="0110110110")
    ans = graph.bfs(max_layer_size_to_explore=10)
    assert not ans.bfs_completed
    assert ans.layer_sizes == [1, 3, 4, 6, 11]


def test_bfs_max_layer_size_to_store():
    graph = CayleyGraph(prepare_graph("lrx", n=10)[0], dest="0110110110")
    ans = graph.bfs(max_layer_size_to_store=10)
    assert ans.bfs_completed
    assert ans.diameter() == 17
    assert ans.layers.keys() == {0, 1, 2, 3, 12, 13, 14, 15, 16, 17}


def test_bfs_start_state():
    graph = CayleyGraph(prepare_graph("lrx", n=5)[0])
    ans = graph.bfs(start_states=[0, 1, 2, 1, 0])
    assert ans.bfs_completed
    assert ans.layer_sizes == [1, 3, 5, 8, 7, 5, 1]


def test_bfs_multiple_start_states():
    graph = CayleyGraph(prepare_graph("lrx", n=5)[0])
    ans = graph.bfs(start_states=[[0, 1, 2, 1, 0], [1, 0, 2, 0, 1], [0, 1, 1, 2, 0]])
    assert ans.bfs_completed
    assert ans.layer_sizes == [3, 9, 11, 6, 1]


@pytest.mark.parametrize("bit_encoding_width", [None, 6])
def test_bfs_lrx_n40_layers5(bit_encoding_width):
    # We need 6*40=240 bits for encoding, so each states is encoded by four int64's.
    n = 40
    generators, dest = prepare_graph("lrx", n=n)
    graph = CayleyGraph(generators, dest=dest, bit_encoding_width=bit_encoding_width)
    assert graph.bfs(max_diameter=5).layer_sizes == [1, 3, 6, 12, 24, 48]


def test_bfs_last_layer_lrx_n8():
    generators, dest = prepare_graph("lrx", n=8)
    graph = CayleyGraph(generators, dest=dest)
    assert graph.bfs().last_layer() == ["10765432"]


def test_bfs_last_layer_lrx_coset_n8():
    generators, _ = prepare_graph("lrx", n=8)
    graph = CayleyGraph(generators, dest="01230123")
    assert set(graph.bfs().last_layer()) == {"11003322", "22110033", "33221100", "00332211"}


@pytest.mark.parametrize("bit_encoding_width", [None, 3, 10, 'auto'])
def test_bfs_bit_encoding(bit_encoding_width):
    generators, _ = prepare_graph("lrx", n=8)
    result = CayleyGraph(generators, bit_encoding_width=bit_encoding_width).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.parametrize("bit_encoding_width", [None, 'auto'])
@pytest.mark.parametrize("batch_size", [100, 1000, 10 ** 9])
def test_bfs_batching(bit_encoding_width, batch_size: int):
    generators, _ = prepare_graph("lrx", n=8)
    result = CayleyGraph(generators, bit_encoding_width=bit_encoding_width, batch_size=batch_size).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.parametrize("hash_chunk_size", [100, 1000, 10 ** 9])
def test_bfs_hash_chunking(hash_chunk_size: int):
    generators, _ = prepare_graph("lrx", n=8)
    result = CayleyGraph(generators, hash_chunk_size=hash_chunk_size).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


def test_free_memory():
    generators, _ = prepare_graph("lrx", n=8)
    result = CayleyGraph(generators, memory_limit_gb=0.0001).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.parametrize("bit_encoding_width", [None, 5])
def test_get_neighbors(bit_encoding_width):
    # Directly check _get_neighbors_batched.
    # In what order it generates neighbours is an implementation detail. However, we rely on this convention when
    # generating the edges list.
    graph = CayleyGraph([[1, 0, 2, 3, 4], [0, 1, 2, 4, 3]], bit_encoding_width=bit_encoding_width)
    states = graph._encode_states(torch.tensor([[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]], dtype=torch.int64))
    result = graph._decode_states(graph._get_neighbors_batched(states))
    if bit_encoding_width == 5:
        # When using StringEncoder, we go over the generators in outer loop, and over the states in inner loop.
        assert torch.equal(result, torch.tensor(
            [[11, 10, 12, 13, 14], [16, 15, 17, 18, 19], [10, 11, 12, 14, 13], [15, 16, 17, 19, 18]]))
    else:
        # When operating on ints directly, it's the other way around.
        assert torch.equal(result, torch.tensor(
            [[11, 10, 12, 13, 14], [10, 11, 12, 14, 13], [16, 15, 17, 18, 19], [15, 16, 17, 19, 18]]))


def test_edges_list_n2():
    graph = CayleyGraph(prepare_graph("lrx", n=2)[0], dest="01")
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.named_undirected_edges() == {('01', '10')}


def test_edges_list_n3():
    graph = CayleyGraph(prepare_graph("lrx", n=3)[0], dest="001")
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.named_undirected_edges() == {('001', '001'), ('001', '010'), ('001', '100'), ('010', '100')}


def test_edges_list_n4():
    graph = CayleyGraph(prepare_graph("top_spin", n=4)[0], dest="0011")
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)
    assert result.named_undirected_edges() == {
        ('0011', '0110'), ('0011', '1001'), ('0011', '1100'), ('0110', '0110'), ('0110', '1100'), ('1001', '1001'),
        ('1001', '1100')}


def test_generators_not_inverse_closed():
    graph = CayleyGraph([[1, 2, 3, 0]])
    assert not graph.generators_inverse_closed
    with pytest.raises(AssertionError):
        graph.bfs()


# Tests below compare growth function for small graphs with stored pre-computed results.
def test_lrx_cayley_growth():
    expected = load_dataset("lrx_cayley_growth")
    for n in range(2, 10):
        generators, _ = prepare_graph("lrx", n=int(n))
        result = CayleyGraph(generators).bfs()
        assert result.layer_sizes == expected[str(n)]


def test_top_spin_cayley_growth():
    expected = load_dataset("top_spin_cayley_growth")
    for n in range(4, 10):
        generators, _ = prepare_graph("top_spin", n=int(n))
        result = CayleyGraph(generators).bfs()
        assert result.layer_sizes == expected[str(n)]


def test_lrx_coset_growth():
    expected = load_dataset("lrx_coset_growth")
    for initial_state in expected.keys():
        if len(initial_state) > 15:
            continue
        generators, _ = prepare_graph("lrx", n=len(initial_state))
        result = CayleyGraph(generators, dest=initial_state).bfs()
        assert result.layer_sizes == expected[initial_state]


def test_top_spin_coset_growth():
    expected = load_dataset("top_spin_coset_growth")
    for initial_state in expected.keys():
        if len(initial_state) > 15:
            continue
        generators, _ = prepare_graph("top_spin", n=len(initial_state))
        result = CayleyGraph(generators, dest=initial_state).bfs()
        assert result.layer_sizes == expected[initial_state]


# To skip slower tests ike this, do `FAST=1 pytest`
@pytest.mark.skipif(FAST_RUN, reason="slow test")
def test_cube222_qtm():
    generators, dest = prepare_graph("cube_2/2/2_6gensQTM")
    graph = CayleyGraph(generators, dest=dest)
    result = graph.bfs()
    assert result.num_vertices == 3674160
    assert result.diameter() == 14
    assert result.layer_sizes == [
        1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276]


@pytest.mark.skipif(FAST_RUN, reason="slow test")
def test_cube222_htm():
    generators, dest = prepare_graph("cube_2/2/2_9gensHTM")
    graph = CayleyGraph(generators, dest=dest)
    result = graph.bfs()
    assert result.num_vertices == 3674160
    assert result.diameter() == 11
    assert result.layer_sizes == [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]


# Below is the benchmark code. To tun: `BENCHMARK=1 pytest . -k benchmark`
@pytest.mark.skipif(not BENCHMARK_RUN, reason="benchmark")
@pytest.mark.parametrize("benchmark_mode", ["baseline", "bit_encoded"])
@pytest.mark.parametrize("n", [28])
def test_benchmark_top_spin(benchmark, benchmark_mode, n):
    generators, _ = prepare_graph("lrx", n=n)
    dest = [0] * (n // 2) + [1] * (n // 2)
    bit_encoding_width = 1 if benchmark_mode == "bit_encoded" else None
    graph = CayleyGraph(generators, dest=dest, bit_encoding_width=bit_encoding_width)
    benchmark(lambda: graph.bfs())
