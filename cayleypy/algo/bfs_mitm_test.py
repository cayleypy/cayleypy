import torch

from cayleypy import PermutationGroups, CayleyGraph, Puzzles
from cayleypy.algo import MeetInTheMiddle


def test_find_path_bfs_mitm_lrx10():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    br12 = graph.bfs(max_diameter=12, return_all_hashes=True)
    br13 = graph.bfs(max_diameter=13, return_all_hashes=True)
    start_state = [7, 9, 6, 1, 0, 8, 5, 3, 2, 4]

    # Too few layers, path not found.
    assert MeetInTheMiddle.find_path_from(graph, start_state, br12) is None

    # To find path of length 26, need minimum of 13 layers in pre-computed BFS.
    path = MeetInTheMiddle.find_path_from(graph, start_state, br13)
    assert path is not None
    assert len(path) == 26
    graph.validate_path(start_state, path)


def test_find_path_bfs_mitm_lrx20():
    graph = CayleyGraph(PermutationGroups.lrx(20))
    br11 = graph.bfs(max_diameter=11, return_all_hashes=True)
    br12 = graph.bfs(max_diameter=12, return_all_hashes=True)
    start_state = [10, 12, 13, 18, 14, 16, 15, 17, 19, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 11]

    # Too few layers, path not found.
    assert MeetInTheMiddle.find_path_from(graph, start_state, br11) is None

    # To find path of length 24, need minimum of 12 layers in pre-computed BFS.
    path = MeetInTheMiddle.find_path_from(graph, start_state, br12)
    assert path is not None
    assert len(path) == 24
    graph.validate_path(start_state, path)


def test_find_path_bfs_mitm_cube222():
    graph = CayleyGraph(Puzzles.rubik_cube(2, metric="fixed_HTM"), verbose=2)
    br = graph.bfs(max_diameter=6, return_all_hashes=True)
    start_state = [0, 0, 4, 0, 0, 1, 5, 4, 2, 2, 5, 1, 3, 3, 5, 2, 4, 1, 5, 3, 2, 3, 1, 4]
    path = MeetInTheMiddle.find_path_from(graph, start_state, br)
    assert len(path) == 11
    graph.validate_path(start_state, path)


def test_mitm_lx10():
    # This test checks that MITM finds paths correctly for not inverse-closed generators (such as LX).
    graph = CayleyGraph(PermutationGroups.lx(10))
    br16 = graph.bfs(max_diameter=17, return_all_hashes=True)
    br17 = graph.bfs(max_diameter=18, return_all_hashes=True)
    dest_state = torch.tensor([7, 9, 6, 1, 0, 8, 5, 3, 2, 4])

    # Too few layers, path not found.
    assert MeetInTheMiddle.find_path_to(graph, dest_state, br16) is None

    # To find path of length 34, need minimum of 17 layers in pre-computed BFS.
    path = MeetInTheMiddle.find_path_to(graph, dest_state, br17)
    assert path is not None
    assert len(path) == 36
    assert torch.equal(graph.apply_path(graph.central_state, path)[0], dest_state)


def test_mitm_find_path_between_lx10():
    graph = CayleyGraph(PermutationGroups.lx(10))
    perm1 = [2, 6, 4, 5, 9, 7, 8, 3, 1, 0]
    perm2 = [7, 1, 0, 6, 2, 4, 5, 9, 8, 3]

    assert MeetInTheMiddle.find_path_between(graph, perm1, perm2, 4) is None
    path = MeetInTheMiddle.find_path_between(graph, perm1, perm2, 5)
    assert path == [1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
