from typing import Optional, Union

import numpy as np
import torch

from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef, MatrixGenerator
from .graphs_lib import prepare_graph


def _to_matrix_generator(m: Union[MatrixGenerator, list, np.ndarray]) -> MatrixGenerator:
    if isinstance(m, MatrixGenerator):
        return m
    return MatrixGenerator.create(m)


def create_graph(
    *,
    generators_permutations: Union[list[list[int]], torch.Tensor, np.ndarray, None] = None,
    generators_matrices: Optional[list[Union[MatrixGenerator, list, np.ndarray]]] = None,
    generator_names: Optional[list[str]] = None,
    name: str = "",
    central_state=None,
    make_inverse_closed: bool = False,
    **kwargs,
) -> CayleyGraph:
    """Creates CayleyGraph.

    All arguments are optional, but one of these must be specified: ``generators_permutations``,
    ``generators_matrices``, ``name``.

    Additional arguments may be passed via kwargs:

    * when creating graph by ``name``, other parameters of the graph (such as "n") that will be passed
      to ``prepare_graph``,
    * any arguments that are accepted by ``CayleyGraph`` constructor (e.g. ``verbose=2``).r.

    This function allows to create graphs in a uniform way. It is useful when you want to specify graph type and
    parameters in a config and have the same code handling different configs.

    This is not recommended in most cases. Instead, create ``CayleyGraphDef`` using one of library classes and then pass
    it to ``CayleyGraph`` constructor.

    :param generators_permutations: List of generating permutations.
    :param generators_matrices: List of generating n*n matrices.
    :param generator_names: Names of the generators.
    :param name: the name of the graph. If generators are not explicitly specified, this will be used to create graph,
        see ``prepare_graph`` source for supported names.
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
        generators = [_to_matrix_generator(g) for g in generators_matrices]
        graph_def = CayleyGraphDef.for_matrix_group(
            generators=generators, generator_names=generator_names, central_state=central_state, name=name
        )
    else:
        assert name != "", "Must specify one of: generators_permutations, generators_matrices or name."
        graph_def = prepare_graph(name, **kwargs)
        if central_state is not None:
            graph_def = graph_def.with_central_state(central_state)
    if make_inverse_closed:
        graph_def = graph_def.make_inverse_closed()
    return CayleyGraph(graph_def, **kwargs)
