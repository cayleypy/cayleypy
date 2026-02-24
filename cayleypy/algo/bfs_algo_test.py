import numpy as np
import pytest

from . import BfsAlgorithm
from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import CayleyGraphDef
from ..datasets import load_dataset
from ..graphs_lib import PermutationGroups


def _layer_to_set(layer: np.ndarray) -> set[str]:
    return set("".join(str(x) for x in state) for state in layer)


def test_bfs_growth_swap():
    graph = CayleyGraph(CayleyGraphDef.create([[1, 0]], central_state="01"))
    result = BfsAlgorithm.bfs(graph)
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


def test_bfs_small_hash_chunk_size():
    graph_def = PermutationGroups.lrx(20)
    graph = CayleyGraph(graph_def, hash_chunk_size=100)
    assert graph.bfs(max_diameter=8).layer_sizes == [1, 3, 6, 12, 24, 48, 91, 172, 325]
