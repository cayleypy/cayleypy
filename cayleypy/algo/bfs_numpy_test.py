from cayleypy import CayleyGraph, load_dataset, bfs_numpy, PermutationGroups


def test_bfs_numpy():
    graph = CayleyGraph(PermutationGroups.lrx(7))
    assert bfs_numpy(graph) == load_dataset("lrx_cayley_growth")["7"]

    graph = CayleyGraph(PermutationGroups.top_spin(7))
    assert bfs_numpy(graph) == load_dataset("top_spin_cayley_growth")["7"]

    graph = CayleyGraph(PermutationGroups.pancake(7))
    assert bfs_numpy(graph) == load_dataset("pancake_cayley_growth")["7"]

    central_state = "000000000111111111"
    graph = CayleyGraph(PermutationGroups.top_spin(18).with_central_state(central_state))
    assert bfs_numpy(graph) == load_dataset("top_spin_coset_growth")[central_state]


# This test checks that StringEncoder.implement_permutation_1d works correctly.
def test_bfs_numpy_lrx_16():
    graph = CayleyGraph(PermutationGroups.lrx(16))
    result = bfs_numpy(graph, max_diameter=10)
    assert result == [1, 3, 6, 12, 24, 48, 91, 172, 324, 596, 1092]
