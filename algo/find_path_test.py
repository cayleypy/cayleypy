import os

import pytest

from cayleypy import create_graph, PermutationGroups, CayleyGraph
from cayleypy import find_path

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


def test_find_path_pancake8():
    graph = CayleyGraph(PermutationGroups.pancake(8))
    start_state = [4, 7, 3, 2, 0, 5, 1, 6]
    path = find_path(graph, start_state)
    assert path is not None
    graph.validate_path(start_state, path)


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
@pytest.mark.parametrize("graph_name", ["lx-9", "lrx-9", "lrx-12", "lrx-15", "lrx-16", "cube_2/2/2_6gensQTM"])
def test_find_path(graph_name: str):
    graph = create_graph(name=graph_name)
    start_state = graph.random_walks(width=1, length=100)[0][-1]
    path = find_path(graph, start_state)
    assert path is not None
    graph.validate_path(start_state, path)
