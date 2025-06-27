from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import torch
from scipy.sparse import coo_array
import h5py
from cayleypy.permutation_utils import apply_permutation
from .cayley_graph_def import CayleyGraphDef


@dataclass(frozen=True)
class BfsResult:
    """Result of running breadth-first search on a Schreier coset graph.

    Can be used to obtain the graph explicitly. In this case, vertices are numbered sequentially in the order in which
    they are visited by BFS.
    """

    bfs_completed: bool  # Whether full graph was explored.
    layer_sizes: list[int]  # i-th element is number of states at distance i from start.
    layers: dict[int, torch.Tensor]  # Explicitly stored states for each layer.

    # Hashes of all vertices (if requested).
    # Order is the same as order of states in layers.
    vertices_hashes: Optional[torch.Tensor]

    # List of edges (if requested).
    # Tensor of shape (num_edges, 2) where vertices are represented by their hashes.
    edges_list_hashes: Optional[torch.Tensor]

    # Definition of the CayleyGraph on which BFS was run. Needed if we want to restore edge names.
    graph: "CayleyGraphDef"

    def diameter(self):
        """Maximal distance from any start vertex to any other vertex."""
        return len(self.layer_sizes) - 1

    def get_layer(self, layer_id: int) -> np.ndarray:
        """Returns all states in the layer with given index."""
        if not 0 <= layer_id <= self.diameter():
            raise KeyError(f"No such layer: {layer_id}.")
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} was not computed because it was too large.")
        return self.layers[layer_id].cpu().numpy()

    def last_layer(self) -> np.ndarray:
        """Returns last layer, formatted as set of strings."""
        return self.get_layer(self.diameter())

    def save(self, path: str):
        assert path.endswith(".h5"), "Please use '.h5' exntention for BfsResult saving"

        with h5py.File(path, "w") as f:
            f["bfs_completed"] = self.bfs_completed
            f["layer_sizes"] = self.layer_sizes
            for name, layer in self.layers.items():
                f[f"layer__{name}"] = layer

            if hasattr(self, "vertices_hashes") and self.vertices_hashes is not None:
                f["vertices_hashes"] = self.vertices_hashes
            else:
                f["vertices_hashes"] = torch.empty([])

            if hasattr(self, "edges_list_hashes") and self.edges_list_hashes is not None:
                f["edges_list_hashes"] = self.edges_list_hashes
            else:
                f["edges_list_hashes"] = torch.empty([])

            f["graph__generators"] = self.graph.generators
            f["graph__generator_names"] = self.graph.generator_names
            f["graph__central_state"] = self.graph.central_state

    @staticmethod
    def load(path: str):
        with h5py.File(path, "r") as f:

            layer_sizes = f["layer_sizes"][()].tolist()

            if f["vertices_hashes"].shape == tuple():
                vertices_hashes = None
            else:
                vertices_hashes = f["vertices_hashes"][()]

            if f["edges_list_hashes"].shape == tuple():
                edges_list_hashes = None
            else:
                edges_list_hashes = f["edges_list_hashes"][()]

            loaded_result = BfsResult(
                bfs_completed=bool(f["bfs_completed"]),
                layer_sizes=layer_sizes,
                layers={x: f[f"layer__{str(x)}"][()] for x in range(len(layer_sizes))},
                edges_list_hashes=edges_list_hashes,
                vertices_hashes=vertices_hashes,
                graph=CayleyGraphDef(
                    generators=f["graph__generators"][()].tolist(),
                    generator_names=f["graph__generator_names"][()].tolist(),
                    central_state=f["graph__central_state"][()].tolist(),
                ),
            )
        return loaded_result

    @cached_property
    def num_vertices(self) -> int:
        """Number of vertices in the graph."""
        return sum(self.layer_sizes)

    @cached_property
    def hashes_to_indices_dict(self) -> dict[int, int]:
        """Dictionary used to remap vertex hashes to indexes."""
        n = self.num_vertices
        assert self.vertices_hashes is not None, "Run bfs with return_all_hashes=True."
        assert len(self.vertices_hashes) == n, "Number of vertices hashes must be the same as the number of veritces"
        ans: dict[int, int] = {}

        for i in range(n):
            ans[int(self.vertices_hashes[i])] = i
        assert len(ans) == n, "Hash collision."
        return ans

    @cached_property
    def edges_list(self) -> np.ndarray:
        """Returns list of edges, with vertices renumbered."""
        assert self.edges_list_hashes is not None, "Run bfs with return_all_edges=True."
        hashes_to_indices = self.hashes_to_indices_dict
        return np.array([[hashes_to_indices[int(h)] for h in row] for row in self.edges_list_hashes], dtype=np.int64)

    def named_undirected_edges(self) -> set[tuple[str, str]]:
        """Names for vertices (representing coset elements in readable format)."""
        vn = self.vertex_names
        return {tuple(sorted([vn[i1], vn[i2]])) for i1, i2 in self.edges_list}  # type: ignore

    def adjacency_matrix(self) -> np.ndarray:
        """Returns adjacency matrix as a dense NumPy array."""
        ans = np.zeros((self.num_vertices, self.num_vertices), dtype=np.int8)
        for i1, i2 in self.edges_list:
            ans[i1, i2] = 1
        return ans

    def adjacency_matrix_sparse(self) -> coo_array:
        """Returns adjacency matrix as a sparse SciPy array."""
        num_edges = len(self.edges_list)
        data = np.ones((num_edges,), dtype=np.int8)
        row = self.edges_list[:, 0]
        col = self.edges_list[:, 1]
        return coo_array((data, (row, col)), shape=(self.num_vertices, self.num_vertices))

    @cached_property
    def vertex_names(self) -> list[str]:
        """Returns names for vertices in the graph."""
        ans = []
        delimiter = "" if max(self.graph.central_state) <= 9 else ","
        for layer_id in range(len(self.layers)):
            if layer_id not in self.layers:
                raise ValueError("To get explicit graph, run bfs with max_layer_size_to_store=None.")
            for state in self.get_layer(layer_id):
                ans.append(delimiter.join(str(int(x)) for x in state))
        return ans

    @cached_property
    def all_states(self) -> torch.Tensor:
        """Explicit states, ordered by index."""
        return torch.vstack([self.layers[i] for i in range(len(self.layer_sizes))])

    def get_edge_name(self, i1: int, i2: int) -> str:
        """Returns name for generator used to go from vertex i1 to vertex i2."""
        state_before = list(map(int, self.all_states[i1]))
        state_after = list(map(int, self.all_states[i2]))
        for i in range(self.graph.n_generators):
            if apply_permutation(self.graph.generators[i], state_before) == state_after:
                return self.graph.generator_names[i]
        assert False, "Edge not found."

    def to_networkx_graph(self, directed=False, with_labels=True):
        """Returns explicit graph as networkx.Graph or networkx.DiGraph."""
        # Import networkx here so we don't need to depend on this library in requirements.
        import networkx  # pylint: disable=import-outside-toplevel

        vertex_names = self.vertex_names
        ans = networkx.DiGraph() if directed else networkx.Graph()
        for name in vertex_names:
            ans.add_node(name)
        for i1, i2 in self.edges_list:
            label = self.get_edge_name(i1, i2) if with_labels else None
            ans.add_edge(vertex_names[i1], vertex_names[i2], label=label)
        return ans
