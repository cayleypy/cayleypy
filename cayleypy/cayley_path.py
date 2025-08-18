from dataclasses import dataclass
from functools import cached_property

import torch

from .cayley_graph import CayleyGraph
from .cayley_graph_def import CayleyGraphDef


@dataclass(frozen=True)
class CayleyPath:
    """Path in a Cayley graph."""

    start_state: torch.Tensor  # First state of the path.
    edges: list[int]  # Edges, represented by generator IDs.
    graph: CayleyGraphDef

    @cached_property
    def all_states(self) -> list[torch.Tensor]:
        """Returns all states on the path."""
        ans = [self.start_state]
        graph = CayleyGraph(self.graph)
        for gen_id in self.edges:
            prev_state = ans[-1]
            next_state = torch.zeros_like(prev_state)
            graph.apply_generator_batched(gen_id, prev_state, next_state)
            ans.append(next_state)
        return ans

    def __repr__(self) -> str:
        return self.graph.path_to_string(self.edges)
