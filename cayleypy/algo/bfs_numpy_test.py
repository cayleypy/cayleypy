from cayleypy import CayleyGraph, load_dataset, bfs_numpy, PermutationGroups


def test_bfs_numpy():
    graph = CayleyGraph(PermutationGroups.lrx(7), bit_encoding_width="auto")
    assert bfs_numpy(graph) == load_dataset("lrx_cayley_growth")["7"]

    graph = CayleyGraph(PermutationGroups.top_spin(7), bit_encoding_width="auto")
    assert bfs_numpy(graph) == load_dataset("top_spin_cayley_growth")["7"]

    graph = CayleyGraph(PermutationGroups.pancake(7), bit_encoding_width="auto")
    assert bfs_numpy(graph) == load_dataset("pancake_cayley_growth")["7"]

    central_state = "000000000111111111"
    graph = CayleyGraph(PermutationGroups.top_spin(18).with_central_state(central_state), bit_encoding_width="auto")
    assert bfs_numpy(graph) == load_dataset("top_spin_coset_growth")[central_state]


# This test checks that StringEncoder.implement_permutation_1d works correctly.
def test_bfs_numpy_lrx_16():
    graph = CayleyGraph(PermutationGroups.lrx(16), bit_encoding_width="auto")
    result = bfs_numpy(graph, max_diameter=10)
    assert result == [1, 3, 6, 12, 24, 48, 91, 172, 324, 596, 1092]


# =============================================================================
# Edge cases
# =============================================================================


def test_bfs_numpy_max_diameter_cap():
    """``max_diameter`` caps the number of BFS layers."""
    graph = CayleyGraph(PermutationGroups.lrx(5), bit_encoding_width="auto")
    result = bfs_numpy(graph, max_diameter=2)
    # Only layers 0, 1, 2 -> 3 elements.
    assert len(result) == 3
    assert result[0] == 1


def test_bfs_numpy_small_graph():
    """BFS on the smallest valid LRX graph (n=3) completes and returns layer sizes."""
    graph = CayleyGraph(PermutationGroups.lrx(3), bit_encoding_width="auto")
    result = bfs_numpy(graph)
    assert result[0] == 1
    assert len(result) >= 1
