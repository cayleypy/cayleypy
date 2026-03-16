import pytest
import torch

from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import CayleyGraphDef
from ..datasets import load_dataset
from ..graphs_lib import PermutationGroups


HAS_CUDA = torch.cuda.is_available()
HAS_MULTI_GPU = HAS_CUDA and torch.cuda.device_count() >= 2


def test_bfs_num_gpus_1_matches_default():
    graph_def = PermutationGroups.lrx(8)
    result = CayleyGraph(graph_def, num_gpus=1).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


def test_bfs_cpu_num_gpus_zero_is_allowed():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu", num_gpus=0)
    assert graph.num_gpus == 0
    assert graph.bfs(max_diameter=3).layer_sizes == load_dataset("lrx_cayley_growth")["5"][:4]


def test_bfs_cpu_rejects_multiple_gpus():
    with pytest.raises(ValueError, match="device='cpu'"):
        CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=2)


@pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA")
def test_bfs_gpu_rejects_zero_gpus():
    with pytest.raises(ValueError, match="num_gpus must be positive"):
        CayleyGraph(PermutationGroups.lrx(5), device="cuda", num_gpus=0)


@pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA")
def test_bfs_gpu_rejects_too_many_gpus():
    with pytest.raises(ValueError, match="only .* are available"):
        CayleyGraph(PermutationGroups.lrx(5), device="cuda", num_gpus=torch.cuda.device_count() + 1)


@pytest.mark.skipif(not HAS_MULTI_GPU, reason="requires >= 2 CUDA GPUs")
def test_bfs_specific_devices_override_device_and_num_gpus():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=1, specific_devices=[1, 0])
    assert graph.device == torch.device("cuda:1")
    assert graph.gpu_devices == [torch.device("cuda:1"), torch.device("cuda:0")]
    assert graph.num_gpus == 2


@pytest.mark.skipif(not HAS_MULTI_GPU, reason="requires >= 2 CUDA GPUs")
def test_bfs_multi_gpu_lrx():
    graph_def = PermutationGroups.lrx(8)
    result = CayleyGraph(graph_def, device="cuda", num_gpus=2).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.skipif(not HAS_MULTI_GPU, reason="requires >= 2 CUDA GPUs")
def test_bfs_multi_gpu_coset():
    graph_def = PermutationGroups.lrx(10).with_central_state("0110110110")
    result = CayleyGraph(graph_def, device="cuda", num_gpus=2).bfs()
    assert result.layer_sizes == [1, 3, 4, 6, 11, 16, 19, 23, 31, 29, 20, 14, 10, 10, 6, 3, 3, 1]


@pytest.mark.skipif(not HAS_MULTI_GPU, reason="requires >= 2 CUDA GPUs")
@pytest.mark.parametrize("batch_size", [100, 1000])
def test_bfs_multi_gpu_batching(batch_size: int):
    graph_def = PermutationGroups.lrx(8)
    result = CayleyGraph(graph_def, device="cuda", num_gpus=2, batch_size=batch_size).bfs()
    assert result.layer_sizes == load_dataset("lrx_cayley_growth")["8"]


@pytest.mark.skipif(not HAS_MULTI_GPU, reason="requires >= 2 CUDA GPUs")
def test_bfs_multi_gpu_not_inverse_closed():
    graph_def = CayleyGraphDef.create([[1, 2, 3, 0]])
    result = CayleyGraph(graph_def, device="cuda", num_gpus=2).bfs()
    assert result.layer_sizes == [1, 1, 1, 1]


@pytest.mark.skipif(not HAS_MULTI_GPU, reason="requires >= 2 CUDA GPUs")
def test_bfs_multi_gpu_modified_copy_preserves_num_gpus():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cuda", num_gpus=2)
    new_graph = graph.modified_copy(graph_def.with_central_state("01210"))
    assert new_graph.num_gpus == 2
