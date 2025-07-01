from dataclasses import dataclass
from typing import Optional
from .cayley_graph_def import CayleyGraphDef


@dataclass(frozen=True)
class BeamSearchResult:
    """Result of running `CayleyGraph.beam_search`."""

    path_found: bool  # Whether full graph was explored.
    path_length: int  # Distance of found path from start state to central state.
    path: Optional[list[int]]  # Path from start state to central state (edges are generator indexes), if requested.
    graph: CayleyGraphDef  # Definition of graph on which beam search was run.

    def get_path_as_string(self):
        assert self.path is not None
        return ",".join(self.graph.generator_names[i] for i in self.path)
