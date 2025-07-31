from typing import Optional, Union

import numpy as np
import torch

from . import CayleyGraphDef, MatrixGenerator
from .cayley_graph import AnyStateType, CayleyGraph
from .graphs_lib import prepare_graph


def create_graph(
    *,
    generators_permutations: Union[list[list[int]], torch.Tensor, np.ndarray, None] = None,
    generators_matrices: Optional[list[Union[MatrixGenerator, list, np.ndarray]]] = None,
    generator_names: Optional[list[str]] = None,
    name: Optional[str] = None,
    central_state: Optional[AnyStateType] = None,
    make_inverse_closed: bool = False,
    **kwargs,
) -> CayleyGraph:
    """Creates CayleyGraph from kwargs.

    Pass the following to kwargs:
        * other parameters of the graph (such as "n") that will be passed to ``prepare_graph``, <-- if name.
        * any arguments that are accepted by ``CayleyGraph`` constructor (e.g. ``verbose=2``).

    All passed kwargs will be first passed to ``prepare_graph`` to construct ``CayleyGraphDef`` and then to
    ``CayleyGraph`` constructor.

    This function allows to create graphs in a uniform way. It is useful when you want to specify graph type and
    parameters in a config and have the same code handling different configs.

    This is not recommended in most cases. Instead, create ``CayleyGraphDef`` using one of library classes and then pass
    it to ``CayleyGraph`` constructor.

    :param generators_permutations: AAAA
    :param generators_matrices: AAAA
    :param generator_names: AAAA
    :param name: the name of the graph, see ``prepare_graph`` source for supported names,
    :param central_state: central state of the graph.
    :param make_inverse_closed: if generators are not inverse-closed and ``make_inverse_closed=True``, adds inverse
        generators to make set of generators inverse-closed.
    :return: created CayleyGraph.
    """
    if generators_permutations is not None:
        assert generators_matrices is None
        graph_def = CayleyGraphDef.create(
            generators_permutations, generator_names=generator_names, central_state=central_state, name=name
        )
    elif generators_matrices is not None:
        assert generators_permutations is None
        generators = [g if g is MatrixGenerator else MatrixGenerator.create(g) for g in generators_matrices]
        graph_def = CayleyGraphDef.for_matrix_group(
            generators=generators, generator_names=generator_names, central_state=central_state, name=name
        )
    else:
        assert name is not None
        graph_def = prepare_graph(name, **kwargs)
    if make_inverse_closed:
        graph_def = graph_def.make_inverse_closed()
    return CayleyGraph(graph_def, **kwargs)
