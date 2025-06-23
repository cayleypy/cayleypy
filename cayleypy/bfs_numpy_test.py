from cayleypy import CayleyGraph, prepare_graph, load_dataset, bfs_numpy, PermutationGroups


def test_bfs_numpy():
    graph = CayleyGraph(PermutationGroups.lrx(7))
    assert bfs_numpy(graph) == load_dataset("lrx_cayley_growth")["7"]

    graph = CayleyGraph(PermutationGroups.top_spin(7))
    assert bfs_numpy(graph) == load_dataset("top_spin_cayley_growth")["7"]

    graph = CayleyGraph(prepare_graph("pancake", n=7))
    assert bfs_numpy(graph) == load_dataset("pancake_cayley_growth")["7"]

    central_state = "000000000111111111"
    graph = CayleyGraph(PermutationGroups.top_spin(18).with_central_state(central_state))
    assert bfs_numpy(graph) == load_dataset("top_spin_coset_growth")[central_state]
