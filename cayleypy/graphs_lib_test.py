import torch

from cayleypy import prepare_graph


def test_all_transpositions():
    graph = prepare_graph("all_transpositions", n=3)
    assert torch.equal(graph.generators, torch.tensor([[1, 0, 2], [2, 1, 0], [0, 2, 1]]))
    assert graph.generator_names == ["(0,1)", "(0,2)", "(1,2)"]

    graph = prepare_graph("all_transpositions", n=20)
    assert graph.n_generators == (20 * 19) // 2


def test_pancake():
    graph = prepare_graph("pancake", n=6)
    assert graph.n_generators == 5
    assert graph.generator_names == ["R1", "R2", "R3", "R4", "R5"]
    assert torch.equal(graph.generators, torch.tensor(
        [[1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5], [3, 2, 1, 0, 4, 5], [4, 3, 2, 1, 0, 5], [5, 4, 3, 2, 1, 0]]
    ))


def test_cubic_pancake():

    graph = prepare_graph("cubic_pancake", n=15, S=1)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R2"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]
    ))
    graph = prepare_graph("cubic_pancake", n=15, S=2)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R3"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]
    ))
    graph = prepare_graph("cubic_pancake", n=15, S=3)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R13"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14]
        ]
    ))
    graph = prepare_graph("cubic_pancake", n=15, S=4)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R14", "R12"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 14],
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 12, 13, 14]
        ]
    ))
    graph = prepare_graph("cubic_pancake", n=15, S=5)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R13", "R2"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]
    ))
    graph = prepare_graph("cubic_pancake", n=15, S=6)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R13", "R3"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
            [2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ]
    ))
    graph = prepare_graph("cubic_pancake", n=15, S=7)
    assert graph.n_generators == 3
    assert graph.generator_names == ["R15", "R13", "R12"]
    assert torch.equal(graph.generators, torch.tensor(
        [
            [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14],
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 12, 13, 14]
        ]
    ))
    
def test_full_reversals():
    graph = prepare_graph("full_reversals", n=4)
    assert graph.n_generators == 6
    assert graph.generator_names == ["R[0..1]", "R[0..2]", "R[0..3]", "R[1..2]", "R[1..3]", "R[2..3]"]
    assert torch.equal(graph.generators, torch.tensor([
        [1, 0, 2, 3], [2, 1, 0, 3], [3, 2, 1, 0], [0, 2, 1, 3], [0, 3, 2, 1], [0, 1, 3, 2]
    ]))


def test_cube333():
    graph = prepare_graph("cube_3/3/3_12gensQTM")
    assert graph.n_generators == 12

    graph = prepare_graph("cube_3/3/3_18gensHTM")
    assert graph.n_generators == 18
