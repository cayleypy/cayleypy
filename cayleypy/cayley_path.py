from dataclasses import dataclass
from functools import cached_property

import torch

from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef


@dataclass(frozen=True)
class CayleyPath:
    """Path in a Cayley graph."""

    start_state: torch.Tensor
    """First state of the path."""

    edges: list[int]
    """Edges, represented by generator IDs."""

    graph: CayleyGraphDef

    @cached_property
    def all_states(self) -> list[torch.Tensor]:
        """All states on the path."""
        ans = [self.start_state]
        graph = CayleyGraph(self.graph)
        for gen_id in self.edges:
            ans.append(graph.apply_path(ans[-1], [gen_id])[0])
        return ans

    @cached_property
    def end_state(self):
        """Last state of the path."""
        return self.all_states[-1]

    def __repr__(self) -> str:
        return self.graph.path_to_string(self.edges)
