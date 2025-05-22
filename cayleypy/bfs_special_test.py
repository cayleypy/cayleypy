from cayleypy import CayleyGraph, prepare_graph, load_dataset, bfs_numpy


def test_bfs_numpy():
    graph = prepare_graph("lrx", n=7)
    assert bfs_numpy(graph) == load_dataset("lrx_cayley_growth")["7"]

    graph = prepare_graph("top_spin", n=7)
    assert bfs_numpy(graph) == load_dataset("top_spin_cayley_growth")["7"]

    graph = prepare_graph("pancake", n=7)
    assert bfs_numpy(graph) == load_dataset("pancake_cayley_growth")["7"]

    dest = "000000000111111111"
    graph = CayleyGraph(prepare_graph("top_spin", n=18).generators, dest=dest)
    assert bfs_numpy(graph) == load_dataset("top_spin_coset_growth")[dest]
