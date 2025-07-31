from cayleypy import PermutationGroups, CayleyGraph
from cayleypy.algo import find_path_bfs_mitm


def test_find_path_bfs_mitm_lrx10():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    br12 = graph.bfs(max_diameter=12, return_all_hashes=True)
    br13 = graph.bfs(max_diameter=13, return_all_hashes=True)
    start_state = [7, 9, 6, 1, 0, 8, 5, 3, 2, 4]

    # Too few layers, path not found.
    result12 = find_path_bfs_mitm(graph, start_state, br12)
    assert result12 is None

    # To find path of length 26, need minimum of 13 layers in pre-computed BFS.
    path = find_path_bfs_mitm(graph, start_state, br13)
    assert path is not None
    assert len(path) == 26
    graph.validate_path(start_state, path)
