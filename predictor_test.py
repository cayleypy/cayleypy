import torch

from .cayley_graph import CayleyGraph
from .graphs_lib import PermutationGroups
from .predictor import Predictor


def test_hamming_predictor():
    graph_def = PermutationGroups.lrx(5).with_central_state("01001")
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, "hamming")
    states = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 2, 3],
            [0, 1, 1, 1, 1],
        ]
    )
    assert torch.equal(predictor(states), torch.tensor([2, 3, 0, 5, 3, 2]))
