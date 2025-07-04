import numpy as np

from .moves import MINI_PYRAMORPHIX_ALLOWED_MOVES, PYRAMINX_MOVES, MEGAMINX_MOVES
from .puzzles import Puzzles
from ..cayley_graph import CayleyGraph
from ..permutation_utils import is_permutation, inverse_permutation


def test_mini_pyramorphix():
    graph = Puzzles.mini_pyramorphix()
    assert graph.n_generators == len(MINI_PYRAMORPHIX_ALLOWED_MOVES)
    assert graph.generator_names == list(MINI_PYRAMORPHIX_ALLOWED_MOVES.keys())
    expected_generators = np.array([MINI_PYRAMORPHIX_ALLOWED_MOVES[k] for k in graph.generator_names])
    assert np.array_equal(graph.generators, expected_generators)
    for gen in graph.generators:
        assert len(gen) == 24
        assert is_permutation(gen)
    identity = list(range(24))
    assert any(gen != identity for gen in graph.generators)
    for gen in graph.generators:
        inverse = inverse_permutation(gen)
        restored = [gen[i] for i in inverse]
        assert restored == list(range(24))
    assert set(graph.generator_names) == set(MINI_PYRAMORPHIX_ALLOWED_MOVES.keys())


def test_pyraminx():
    perm_set_length = 36
    graph = Puzzles.pyraminx()
    assert graph.n_generators == len(PYRAMINX_MOVES) * 2  # inverse generators are not listed in PYRAMINX_MOVES

    graph_gens = dict(zip(graph.generator_names, graph.generators))
    gen_names = list(PYRAMINX_MOVES.keys())
    gen_names += [x + "_inv" for x in PYRAMINX_MOVES]

    for gen_name, gen in PYRAMINX_MOVES.items():
        assert np.all(graph_gens[gen_name] == gen)
        assert np.all(graph_gens[gen_name + "_inv"] == inverse_permutation(gen))
        assert len(gen) == perm_set_length


def test_megaminx():
    perm_set_length = 120
    graph = Puzzles.megaminx()
    assert graph.n_generators == len(MEGAMINX_MOVES) * 2  # inverse generators are not listed in MEGAMINX_MOVES

    graph_gens = dict(zip(graph.generator_names, graph.generators))
    gen_names = list(MEGAMINX_MOVES.keys())
    gen_names += [x + "_inv" for x in MEGAMINX_MOVES]

    for gen_name, gen in MEGAMINX_MOVES.items():
        assert np.all(graph_gens[gen_name] == gen)
        assert np.all(graph_gens[gen_name + "_inv"] == inverse_permutation(gen))
        assert len(gen) == perm_set_length

    graph = CayleyGraph(graph, device="cpu")
    assert graph.bfs(max_diameter=4).layer_sizes == [1, 24, 408, 6208, 90144]
