import numpy as np

from cayleypy.graphs_lib import prepare_graph


def test_adjacency_matrix():
    graph = prepare_graph("lrx", n=4)
    result = graph.bfs(return_all_edges=True, return_all_hashes=True)

    adj_mx_1 = result.adjacency_matrix()
    assert adj_mx_1.shape == (24, 24)
    assert np.sum(adj_mx_1) == 24 * 3
    assert np.array_equal(adj_mx_1, adj_mx_1.T)
    adj_mx_2 = result.adjacency_matrix_sparse().toarray()
    assert np.array_equal(adj_mx_1, adj_mx_2)
