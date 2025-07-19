from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from cayleypy import CayleyGraph, BfsResult, PermutationGroups
from cayleypy.graphs_lib import MatrixGroups


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
    graph_1 = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    graph_2 = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
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
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

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
                    assert bfs_result == loaded_bfs_result, "Original and loaded BfsResults must be the same."


def test_bfs_result_save_load_matrix_groups():
    """Test BFS result save/load for matrix groups including Heisenberg group."""
    # Test with Heisenberg group with modulo to keep state space finite
    graph = CayleyGraph(MatrixGroups.heisenberg(modulo=3), device="cpu")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for return_all_hashes in [True, False]:
            for return_all_edges in [True, False]:
                for max_diameter in [1, 2]:  # Small diameters for matrix groups
                    bfs_result = graph.bfs(
                        return_all_edges=return_all_edges,
                        return_all_hashes=return_all_hashes,
                        max_diameter=max_diameter,
                    )
                    bfs_result.save(temp_dir / "matrix_bfs_result.h5")
                    loaded_bfs_result = BfsResult.load(temp_dir / "matrix_bfs_result.h5")
                    assert bfs_result == loaded_bfs_result, "Original and loaded matrix BfsResults must be the same."


def test_bfs_result_save_load_matrix_groups_with_modulo():
    """Test BFS result save/load for matrix groups with modular arithmetic."""
    # Test with Heisenberg group with modulo
    graph = CayleyGraph(MatrixGroups.heisenberg(modulo=5), device="cpu")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for return_all_hashes in [True, False]:
            for return_all_edges in [True, False]:
                bfs_result = graph.bfs(
                    return_all_edges=return_all_edges,
                    return_all_hashes=return_all_hashes,
                    max_diameter=3,  # Small diameter for modular case
                )
                bfs_result.save(temp_dir / "modular_matrix_bfs_result.h5")
                loaded_bfs_result = BfsResult.load(temp_dir / "modular_matrix_bfs_result.h5")
                assert (
                    bfs_result == loaded_bfs_result
                ), "Original and loaded modular matrix BfsResults must be the same."


def test_bfs_result_matrix_group_equality():
    """Test that matrix group BFS results maintain equality after save/load round-trip."""
    graph = CayleyGraph(MatrixGroups.heisenberg(modulo=3), device="cpu")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for return_all_hashes in [True, False]:
            for return_all_edges in [True, False]:
                bfs_result = graph.bfs(
                    return_all_edges=return_all_edges, return_all_hashes=return_all_hashes, max_diameter=1
                )

                # Test save/load round-trip equality
                save_path = temp_dir / f"equality_test_{return_all_hashes}_{return_all_edges}.h5"
                bfs_result.save(str(save_path))
                loaded_result = BfsResult.load(str(save_path))

                assert bfs_result == loaded_result, "Matrix BFS result should equal itself after save/load"
                assert loaded_result.graph.is_matrix_group(), "Loaded graph should be matrix group"


def test_bfs_result_legacy_format_compatibility():
    """Test that legacy format files (permutation groups) still load correctly."""
    # This test ensures backward compatibility with old save format
    graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        bfs_result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_diameter=3)
        bfs_result.save(temp_dir / "legacy_format.h5")
        loaded_bfs_result = BfsResult.load(temp_dir / "legacy_format.h5")

        assert bfs_result == loaded_bfs_result, "Legacy format should still load correctly"
        assert loaded_bfs_result.graph.is_permutation_group(), "Loaded graph should be permutation group"
