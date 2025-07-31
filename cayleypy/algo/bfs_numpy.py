"""Special BFS algorithms, optimized for low memory usage."""

import numpy as np

from ..cayley_graph import CayleyGraph
from ..permutation_utils import inverse_permutation


def bfs_numpy(graph: CayleyGraph, max_diameter: int = 1000000) -> list[int]:
    """Simple version of BFS (from destination_state) using numpy, optimized for memory usage."""
    assert graph.definition.is_permutation_group(), "Only works for permutations."
    assert graph.definition.generators_inverse_closed, "Only supports undirected graph."
    assert graph.string_encoder is not None
    assert graph.string_encoder.encoded_length == 1, "Only works on states encoded by single int64."
    perms = graph.definition.generators_permutations
    perm_funcs = [graph.string_encoder.implement_permutation_1d(p) for p in perms]
    pn = len(perms)
    start_state_tensor = graph.encode_states(graph.central_state).cpu().numpy().reshape(-1)
    start_state = np.array(start_state_tensor, dtype=np.int64)

    # For each generating permutation store which one is its inverse.
    inv_perm_idx = []
    for i in range(pn):
        inv = [j for j in range(pn) if inverse_permutation(perms[i]) == perms[j]]
        assert len(inv) == 1
        inv_perm_idx.append(inv[0])

    def _make_states_unique(layer):
        for i1 in range(pn):
            for i2 in range(i1 + 1, pn):
                layer[i1] = np.setdiff1d(layer[i1], layer[i2], assume_unique=True)

    layer0 = [start_state] * pn
    layer1 = [perm_funcs[i](start_state) for i in range(pn)]
    layer1 = [np.setdiff1d(x, start_state, assume_unique=True) for x in layer1]
    _make_states_unique(layer1)
    layer_sizes = [1, len(np.unique(np.hstack(layer1)))]

    for i in range(2, max_diameter + 1):
        layer2 = []
        for i1 in range(pn):
            # All states where we can go from layer1 by permutation i1 (except those that are in layer0).
            next_group = [perm_funcs[i1](layer1[i2]) for i2 in range(pn) if i2 != inv_perm_idx[i1]]
            states = np.hstack(next_group)
            states = np.sort(states)
            for i2 in range(pn):
                states = np.setdiff1d(states, layer0[i2], assume_unique=True)
                states = np.setdiff1d(states, layer1[i2], assume_unique=True)
            layer2.append(states)
        _make_states_unique(layer2)
        layer2_size = sum(len(x) for x in layer2)
        if layer2_size == 0:
            break
        layer_sizes.append(layer2_size)
        if graph.verbose >= 2:
            print(f"Layer {i}: {layer2_size} states.")
        layer0, layer1 = layer1, layer2
        if layer2_size >= 10**9:
            graph.free_memory()

    return layer_sizes
