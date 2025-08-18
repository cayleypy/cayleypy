from typing import Optional

import torch

from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import AnyStateType
from ..torch_utils import isin_via_searchsorted


class InteractiveBfs:
    """Interactive breadth-first search that computes layers one by one."""

    def __init__(self, graph: CayleyGraph, start_states: AnyStateType):
        self.graph = graph
        self.cur_layer = graph.encode_states(start_states)
        self.hashes = [graph.hasher.make_hashes(self.cur_layer)]

    def _remove_seen_states(self, hashes: torch.Tensor) -> torch.Tensor:
        """Returns mask where 0s are at positions in `current_layer_hashes` that were seen previously."""
        ans = ~isin_via_searchsorted(hashes, self.hashes[-1])
        if len(self.hashes) > 1:
            ans &= ~isin_via_searchsorted(hashes, self.hashes[-2])
        if not self.graph.definition.generators_inverse_closed:
            for h in self.hashes[:-2]:
                ans &= ~isin_via_searchsorted(hashes, h)
        return ans

    def step(self):
        """Computes next layer of BFS."""
        next_layer, next_layer_hashes = self.graph.get_unique_states(self.graph.get_neighbors(self.cur_layer))
        mask = self._remove_seen_states(next_layer_hashes)
        self.cur_layer = next_layer[mask]
        self.hashes.append(next_layer_hashes[mask])

    def find_on_last_layer(self, hashes: torch.Tensor) -> Optional[torch.Tensor]:
        """Checks if ``hashes`` intersects with the last layer.

        If there is intersection, returns a state from intersection. Otherwise, returns ``None``.
        """
        mask = isin_via_searchsorted(self.hashes[-1], hashes)
        if not torch.any(mask):
            return None
        return self.graph.decode_states(self.cur_layer[mask.nonzero().reshape((-1,))])[0]
