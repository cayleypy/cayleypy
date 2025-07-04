from .globe import globe_puzzle
from ..cayley_graph_def import CayleyGraphDef
from ..permutation_utils import inverse_permutation, compose_permutations
from .cube import rubik_cube_qstm, rubik_cube_htm, rubik_cube_qtm
from .globe import globe_puzzle


class Puzzles:
    """Definitions of graphs describing various puzzles."""

    @staticmethod
    def rubik_cube(cube_size: int, metric: str) -> CayleyGraphDef:
        """Creates Cayley graph for NxNxN Rubik's cube.

        :param cube_size: - Size of the cube (N).
        :param metric: - metric defining what counts as one move, one of:
          - "QSTM" - Quarter Slice Turn Metric.
          - "QTM": Quarter Turn Metric (only supported for 2x2x2 and 3x3x3).
          - "HTM": Half Turn Metric (only supported for 2x2x2 and 3x3x3).
        """
        if metric == "QSTM":
            return rubik_cube_qstm(cube_size)
        elif metric == "QTM":
            return rubik_cube_qtm(cube_size)
        elif metric == "HTM":
            return rubik_cube_htm(cube_size)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def globe_puzzle(a: int, b: int) -> CayleyGraphDef:
        """Cayley graph for Globe puzzle group, a + 1 cycle and 2b order 2 generators."""
        return globe_puzzle(a, b)
