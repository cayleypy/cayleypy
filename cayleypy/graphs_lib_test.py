import torch

from cayleypy import prepare_graph


def test_all_transpositions():
    graph = prepare_graph("all_transpositions", n=20)
    assert graph.n_generators == (20 * 19) // 2


def test_pancake():
    graph = prepare_graph("pancake", n=6)
    assert graph.n_generators == 5
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5"]
    assert torch.equal(graph.generators[2], torch.tensor([3, 2, 1, 0, 4, 5]))


def test_cube333():
    graph = prepare_graph("cube_3/3/3_12gensQTM")
    assert graph.n_generators == 12

    graph = prepare_graph("cube_3/3/3_18gensHTM")
    assert graph.n_generators == 18
