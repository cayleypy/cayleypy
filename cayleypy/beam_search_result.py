from dataclasses import dataclass
from typing import Optional
from .cayley_graph_def import CayleyGraphDef


@dataclass(frozen=True)
class BeamSearchResult:
    """Result of running `CayleyGraph.beam_search`."""

    path_found: bool  # Whether full graph was explored.
    path_length: int  # Distance of found path from start state to central state.
    path: Optional[list[int]]  # Path from start state to central state (edges are generator indexes), if requested.
    debug_scores: dict[int, float]  # Scores achieved on each step.
    graph: CayleyGraphDef  # Definition of graph on which beam search was run.

    def __post_init__(self):
        if self.path is not None:
            assert len(self.path) == self.path_length

    def get_path_as_string(self, delimiter="."):
        assert self.path is not None
        return self.graph.path_to_string(self.path, delimiter)

    def __repr__(self):
        if not self.path_found:
            return "BeamSearchResult(path_found=False)"
        ans = f"BeamSearchResult(path_length={self.path_length}"
        if self.path is not None and self.path_length > 0:
            ans += ", path=" + self.get_path_as_string()
        ans += ")"
        return ans
