import torch

from cayleypy import prepare_graph


def test_all_transpositions():
    graph = prepare_graph("all_transpositions", n=3)
    assert torch.equal(graph.generators.cpu(), torch.tensor([[1, 0, 2], [2, 1, 0], [0, 2, 1]]))
    assert graph.generator_names == ["(0,1)", "(0,2)", "(1,2)"]

    graph = prepare_graph("all_transpositions", n=20)
    assert graph.n_generators == (20 * 19) // 2


def test_pancake():
    graph = prepare_graph("pancake", n=6)
    assert graph.n_generators == 5
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5"]
    assert torch.equal(graph.generators.cpu(), torch.tensor(
        [[1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [3, 2, 1, 0, 4, 5], [4, 3, 2, 1, 0, 5], [5, 4, 3, 2, 1, 0]]
    ))


def test_burnt_pancake():
    graph = prepare_graph("burnt_pancake", n=6)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert torch.equal(graph.generators.cpu(), torch.tensor(
        [[6, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11],
         [7, 6, 2, 3, 4, 5, 1, 0, 8, 9, 10, 11],
         [8, 7, 6, 3, 4, 5, 2, 1, 0, 9, 10, 11],
         [9, 8, 7, 6, 4, 5, 3, 2, 1, 0, 10, 11],
         [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11],
         [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    ))


def test_full_reversals():
    graph = prepare_graph("full_reversals", n=4)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R[0..1]", "R[0..2]", "R[0..3]", "R[1..2]", "R[1..3]", "R[2..3]"]
    assert torch.equal(graph.generators.cpu(), torch.tensor([
        [1, 0, 2, 3], [2, 1, 0, 3], [3, 2, 1, 0], [0, 2, 1, 3], [0, 3, 2, 1], [0, 1, 3, 2]
    ]))


def test_cube333():
    graph = prepare_graph("cube_3/3/3_12gensQTM")
    assert graph.n_generators == 12

    graph = prepare_graph("cube_3/3/3_18gensHTM")
    assert graph.n_generators == 18
