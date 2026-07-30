from pathlib import Path
from tempfile import TemporaryDirectory
import h5py
import numpy as np
import pytest
from cayleypy import CayleyGraph, BfsResult, PermutationGroups


def test_adjacency_matrix():
    graph = CayleyGraph(PermutationGroups.lrx(4))
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)

    adj_mx_1 = result.adjacency_matrix()
    assert adj_mx_1.shape == (24, 24)
    assert np.sum(adj_mx_1) == 24 * 3
    assert np.array_equal(adj_mx_1, adj_mx_1.T)
    adj_mx_2 = result.adjacency_matrix_sparse().toarray()
    assert np.array_equal(adj_mx_1, adj_mx_2)


def test_bfs_result_eq():
    graph_1 = CayleyGraph(PermutationGroups.lrx(4), device="cpu", random_seed=43)
    graph_2 = CayleyGraph(PermutationGroups.lrx(4), device="cpu", random_seed=43)
    graph_3 = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    for return_all_hashes in [True, False]:
        for return_all_edges in [True, False]:
            bfs_res_1 = graph_1.bfs(return_all_edges=return_all_edges, return_all_hashes=return_all_hashes)
            bfs_res_2 = graph_2.bfs(return_all_edges=return_all_edges, return_all_hashes=return_all_hashes)
            bfs_res_3 = graph_3.bfs(return_all_edges=return_all_edges, return_all_hashes=return_all_hashes)
            bfs_res_3_md = graph_2.bfs(
                return_all_edges=return_all_edges, return_all_hashes=return_all_hashes, max_diameter=2
            )
            # pylint: disable=R0124
            assert bfs_res_1 == bfs_res_1, "Bfs results must be equal for the same graph"
            # pylint: enable=R0124
            assert bfs_res_1 == bfs_res_2, "Bfs results must be equal for the the equal graphs"
            assert bfs_res_1 != bfs_res_3, "Bfs results must be different for different graphs"
            assert bfs_res_3 != bfs_res_3_md, "Bfs results must be different if bfs was not completed"


def test_bfs_result_save_load():

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        for graph_def in [PermutationGroups.lrx(4), PermutationGroups.transposons(8)]:
            graph = CayleyGraph(graph_def, device="cpu")
            for return_all_hashes in [True, False]:
                for return_all_edges in [True, False]:
                    for max_diameter in [3, 1000000]:
                        bfs_result = graph.bfs(
                            return_all_edges=return_all_edges,
                            return_all_hashes=return_all_hashes,
                            max_diameter=max_diameter,
                        )
                        bfs_result.save(temp_dir / "bfs_result.h5")
                        loaded_bfs_result = BfsResult.load(temp_dir / "bfs_result.h5")
                        assert bfs_result.graph == loaded_bfs_result.graph
                        assert bfs_result == loaded_bfs_result, "Original and loaded BfsResults must be the same."


def test_bfs_result_save_load_after_load_methods_work():
    """A5 regression: after BfsResult.load(), tensor methods must work (not numpy)."""
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
        bfs_result = graph.bfs(return_all_hashes=True, max_diameter=3)
        bfs_result.save(temp_dir / "bfs_result.h5")
        loaded = BfsResult.load(temp_dir / "bfs_result.h5")
        # These must not raise AttributeError: 'numpy.ndarray' object has no attribute 'cpu'
        layer = loaded.get_layer(0)
        assert isinstance(layer, np.ndarray)  # get_layer returns np.ndarray (intentional)
        loaded.to_device("cpu")
        last = loaded.last_layer()
        assert isinstance(last, np.ndarray)


def test_bfs_result_load_incomplete_hashes_raises():
    """A5 regression: loading a BFS save with incomplete hash layers must raise."""
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
        bfs_result = graph.bfs(return_all_hashes=True, max_diameter=3)
        save_path = temp_dir / "bfs_result.h5"
        bfs_result.save(save_path)
        # Corrupt the file by removing one hash layer
        with h5py.File(save_path, "a") as f:
            # Find the last edges_list_hashes__ key and delete it
            keys_to_del = [k for k in f.keys() if k.startswith("edges_list_hashes__")]
            if keys_to_del:
                del f[keys_to_del[-1]]
        with pytest.raises(ValueError, match="BFS was saved with hashes but only"):
            BfsResult.load(save_path)
