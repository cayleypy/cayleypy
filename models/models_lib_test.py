import torch

from .models_lib import PREDICTOR_MODELS
from .. import prepare_graph, Predictor, CayleyGraph


def test_loads_predictor_models():
    # Checks that all models can be loaded and successfully return prediction for central state of the graph.
    # This test does not check model quality.
    for graph_name in PREDICTOR_MODELS:
        graph_def = prepare_graph(graph_name)
        graph = CayleyGraph(graph_def)
        predictor = Predictor.pretrained(graph)
        ans = predictor(torch.tensor(graph_def.central_state).reshape((1, -1)))
        assert ans.shape == (1,)
