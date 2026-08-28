import torch

from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef, MatrixGenerator


def _matrix_graph(*, random_seed=None) -> CayleyGraph:
    graph_def = CayleyGraphDef.for_matrix_group(generators=[MatrixGenerator.create([[1, 1], [0, 1]], modulo=5)])
    return CayleyGraph(graph_def, device="cpu", random_seed=random_seed)


def test_explicit_zero_seed_is_preserved(monkeypatch):
    monkeypatch.setattr("cayleypy.hasher.random.randint", lambda *_: 17)

    graph = _matrix_graph(random_seed=0)

    assert graph.hasher.seed == 0


def test_torchrun_ranks_use_the_same_default_hash_seed(monkeypatch):
    generated_seeds = iter([11, 22])
    monkeypatch.setattr("cayleypy.hasher.random.randint", lambda *_: next(generated_seeds))
    monkeypatch.setenv("WORLD_SIZE", "2")

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    rank0_graph = _matrix_graph()

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    rank1_graph = _matrix_graph()

    states = torch.tensor([[1, 0, 0, 1], [1, 1, 0, 1]], dtype=torch.int64)
    assert rank0_graph.hasher.seed == rank1_graph.hasher.seed
    assert torch.equal(rank0_graph.hasher.make_hashes(states), rank1_graph.hasher.make_hashes(states))
