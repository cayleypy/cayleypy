"""Sources of training data for models estimating distances in a Cayley graph.

A data source generates states with target distances. Different sources know different things about the graph: random
walks are cheap but their targets are upper estimates, breadth-first search gives exact targets but only near the
central state, and paths found by beam search give targets for states nothing else reaches. Mixing them is what makes
training work, and :class:`MixtureDataSource` does that in given proportions.

Example:

>>> from cayleypy import CayleyGraph, PermutationGroups
>>> from cayleypy.train import BfsAnchors
>>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
>>> data = BfsAnchors(graph, depth=2).generate()
>>> sorted(int(target) for target in data.targets)
[0, 1, 1, 1, 2, 2, 2, 2, 2]
"""

import abc
import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from ..cayley_path import CayleyPath
from ..models.checkpoint import PathType

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph

# Mode of random walk generation used to produce targets for Q-models, see :class:`SparseQSampler`.
_SPARSE_Q_RANDOM_WALK_MODE = "classic"

# Name of the column with solutions in files published in the "cayleypy-beam-results" repository.
_DEFAULT_SOLUTION_COLUMN = "solution"

# Characters that spreadsheets add in front of a cell to keep it from being interpreted as something else.
_CELL_PREFIXES = "'\"= "


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True)
class TrainingData:
    """States with target distances to train on.

    :param states: States in decoded representation, of shape ``[n_states, state_size]``.
    :param targets: Target distances. Shape is ``[n_states]`` for a model with one output, or
        ``[n_states, n_generators]`` for a Q-model, in which case ``targets[i][j]`` is the target distance of the state
        obtained by applying j-th generator to ``states[i]``.
    :param mask: Which targets are known (optional), of the same shape as `targets`. Unknown targets are ignored by the
        loss, see :class:`cayleypy.train.Loss`. None means that all targets are known.
    :param weights: Importance of every target (optional), of the same shape as `targets`. None means that all targets
        are equally important. Sources whose targets are upper bounds rather than exact distances set this, so that the
        loss can trust them less.
    """

    states: torch.Tensor
    targets: torch.Tensor
    mask: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.states.dim() != 2:
            raise ValueError(f"states must have shape [n_states, state_size], got {tuple(self.states.shape)}.")
        if self.targets.dim() not in (1, 2):
            raise ValueError(
                f"targets must have shape [n_states] or [n_states, n_outputs], got {tuple(self.targets.shape)}."
            )
        if self.targets.shape[0] != self.states.shape[0]:
            raise ValueError(f"There are {self.states.shape[0]} states, but {self.targets.shape[0]} targets for them.")
        for name, tensor in (("mask", self.mask), ("weights", self.weights)):
            if tensor is not None and tensor.shape != self.targets.shape:
                raise ValueError(
                    f"Shape of {name} is {tuple(tensor.shape)}, but it must be the same as shape of targets, which is "
                    f"{tuple(self.targets.shape)}."
                )

    def __len__(self) -> int:
        """Number of states in this data."""
        return int(self.states.shape[0])

    @property
    def n_outputs(self) -> int:
        """Number of model outputs these targets are for (1 if the model predicts a single distance)."""
        return 1 if self.targets.dim() == 1 else int(self.targets.shape[1])

    def select(self, index: torch.Tensor) -> "TrainingData":
        """Returns the part of this data with given indexes of states.

        :param index: Indexes of states to keep.
        :return: Data containing only those states, with their targets.
        """
        return TrainingData(
            states=self.states[index],
            targets=self.targets[index],
            mask=None if self.mask is None else self.mask[index],
            weights=None if self.weights is None else self.weights[index],
        )

    @staticmethod
    def concat(parts: Sequence["TrainingData"]) -> "TrainingData":
        """Concatenates several pieces of data, which must have targets of the same shape.

        Where one piece has a mask (or weights) and another does not, the one without gets a mask of "everything is
        known" (or weights of 1), so that mixing labeled data with sparsely labeled data means what it looks like.

        :param parts: Pieces of data to concatenate. Must be non-empty.
        :return: All of the data in one piece.
        """
        if len(parts) == 0:
            raise ValueError("Cannot concatenate an empty list of training data.")
        n_outputs = parts[0].n_outputs
        for part in parts[1:]:
            if part.n_outputs != n_outputs:
                raise ValueError(
                    f"Cannot concatenate data with targets for {n_outputs} outputs and data with targets for "
                    f"{part.n_outputs} outputs."
                )
        has_mask = any(part.mask is not None for part in parts)
        has_weights = any(part.weights is not None for part in parts)
        targets = [part.targets.to(torch.float32) for part in parts]
        return TrainingData(
            states=torch.cat([part.states for part in parts], dim=0),
            targets=torch.cat(targets, dim=0),
            mask=_cat_or_none([part.mask for part in parts], targets, has_mask, torch.bool),
            weights=_cat_or_none([part.weights for part in parts], targets, has_weights, torch.float32),
        )


def _cat_or_none(
    tensors: Sequence[Optional[torch.Tensor]],
    targets: Sequence[torch.Tensor],
    needed: bool,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Concatenates optional per-piece tensors, filling missing ones with ones of the shape of the targets."""
    if not needed:
        return None
    parts = [
        torch.ones_like(target, dtype=dtype) if tensor is None else tensor.to(dtype)
        for tensor, target in zip(tensors, targets)
    ]
    return torch.cat(parts, dim=0)


def _sample_index(n_available: int, size: Optional[int], device: torch.device) -> torch.Tensor:
    """Returns indexes of `size` states sampled with replacement, or of all of them if `size` is None."""
    if size is None:
        return torch.arange(n_available, device=device)
    if n_available == 0:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    return torch.randint(0, n_available, (size,), device=device)


class DataSource(abc.ABC):
    """Base class for sources of training data.

    A source is created once and generates a fresh piece of data every time :meth:`generate` is called - normally once
    per epoch, so that the model rarely sees the same state twice. How much data one call generates is configured when
    the source is created.
    """

    @abc.abstractmethod
    def generate(self) -> TrainingData:
        """Generates one piece of training data.

        :return: States with target distances.
        """


class RandomWalksSource(DataSource):
    """Training data from random walks, where the number of steps is the target distance.

    This is the cheapest source of data and the only one that reaches distant states, but its targets are upper
    estimates of the true distance (a walk can wander around and come back), which is why the fast-mixing "nbt" mode is
    the default.

    Example:

    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.train import RandomWalksSource
    >>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    >>> len(RandomWalksSource(graph, n_walks=8, rw_length=5).generate())
    40
    """

    def __init__(
        self,
        graph: "CayleyGraph",
        n_walks: int = 128,
        rw_length: int = 20,
        mode: str = "nbt",
        nbt_history_depth: int = 1,
    ):
        """Initializes RandomWalksSource.

        :param graph: Graph to generate walks on.
        :param n_walks: Number of walks to generate (their `width`).
        :param rw_length: Length of every walk.
        :param mode: Mode of random walk generation - see :class:`cayleypy.algo.RandomWalksGenerator`.
        :param nbt_history_depth: For "nbt" mode, how many previous levels to remember and ban from revisiting.
        """
        _check_positive("n_walks", n_walks)
        _check_positive("rw_length", rw_length)
        self.graph = graph
        self.n_walks = int(n_walks)
        self.rw_length = int(rw_length)
        self.mode = mode
        self.nbt_history_depth = int(nbt_history_depth)

    def generate(self) -> TrainingData:
        """Generates random walks with their estimated distances.

        :return: States on the walks, with the number of steps at which they were visited as their target distance.
        """
        states, distances = self.graph.random_walks(
            width=self.n_walks,
            length=self.rw_length,
            mode=self.mode,
            nbt_history_depth=self.nbt_history_depth,
        )
        return TrainingData(states=states, targets=distances.to(torch.float32))


class SparseQSampler(DataSource):
    """Training data for Q-models from random walks, where only the moves along the walk are labeled.

    A Q-model predicts the distance of every child of a state, i.e. it has one output per generator, and a random walk
    says something about exactly two of them. For a state visited at step ``p`` of a walk, the move that leads back to
    the previous state gives a target of ``p-1``, and the move that leads to the next state gives a target of ``p+1``;
    all the other outputs are unlabeled and are masked out, so they contribute to neither the loss nor the gradient.

    Moves are found by comparing children of every state with the previous and the next state on the walk, so if
    several generators lead to the same state, all of them are labeled. Walks are generated in "classic" mode: unlike
    "nbt" and "bfs", it produces actual paths, which is what makes "the previous state on the walk" meaningful.

    Example:

    >>> import torch
    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.train import SparseQSampler
    >>> _ = torch.manual_seed(0)
    >>> graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    >>> data = SparseQSampler(graph, n_walks=4, rw_length=6).generate()
    >>> tuple(data.targets.shape)
    (24, 3)
    >>> int(data.mask[:4].sum())  # First states of the walks have only the forward move labeled.
    4
    """

    def __init__(self, graph: "CayleyGraph", n_walks: int = 128, rw_length: int = 20):
        """Initializes SparseQSampler.

        :param graph: Graph to generate walks on. Its generators must be inverse closed, otherwise the move back to
            the previous state on a walk is not a move of the graph and could not be labeled.
        :param n_walks: Number of walks to generate (their `width`).
        :param rw_length: Length of every walk. Must be at least 2.
        """
        _check_positive("n_walks", n_walks)
        if rw_length < 2:
            raise ValueError(f"rw_length must be at least 2, got {rw_length}.")
        if not graph.definition.generators_inverse_closed:
            raise ValueError(
                "Generators of this graph are not inverse closed, so the move from a state back to the previous state "
                "on a walk is not a move of this graph and cannot be labeled. Consider training on the graph returned "
                "by CayleyGraph.with_inverted_generators()."
            )
        self.graph = graph
        self.n_walks = int(n_walks)
        self.rw_length = int(rw_length)

    def generate(self) -> TrainingData:
        """Generates random walks with sparse targets for the moves along them.

        :return: States on the walks, with targets of shape ``[n_states, n_generators]`` and a mask where only the
            moves to the previous and to the next state on the walk are labeled.
        """
        graph = self.graph
        width, length = self.n_walks, self.rw_length
        n_generators = graph.definition.n_generators
        states, _ = graph.random_walks(width=width, length=length, mode=_SPARSE_Q_RANDOM_WALK_MODE)
        encoded_states = graph.encode_states(states)
        n_states = int(encoded_states.shape[0])

        # i-th walk visits states[i], states[i+width], states[i+2*width], ... so shifting by `width` gives the previous
        # and the next state on the same walk (the states that wrap around are excluded by has_previous/has_next).
        state_hashes = graph.hasher.make_hashes(encoded_states)
        previous_hashes = torch.roll(state_hashes, width).unsqueeze(1)
        next_hashes = torch.roll(state_hashes, -width).unsqueeze(1)
        children_hashes = graph.hasher.make_hashes(graph.get_neighbors(encoded_states))
        children_hashes = children_hashes.reshape((n_generators, n_states)).transpose(0, 1)

        steps = torch.arange(length, device=encoded_states.device).repeat_interleave(width)
        has_previous = (steps > 0).unsqueeze(1)
        has_next = (steps < length - 1).unsqueeze(1)
        leads_back = (children_hashes == previous_hashes) & has_previous
        leads_forward = (children_hashes == next_hashes) & has_next
        steps = steps.unsqueeze(1).to(torch.float32)
        # Where a move leads both back and forward (the walk returned to where it was), the smaller target is right.
        targets = torch.where(leads_back, steps - 1, steps + 1)
        return TrainingData(states=states, targets=targets, mask=leads_back | leads_forward)


class BfsAnchors(DataSource):
    """Training data with exact distances of states near the central state, computed by breadth-first search.

    States close to the central state are the ones a model sees least often in random walks, and mispredicting them is
    expensive: a beam search that is two moves away from the goal and does not see it wanders off. This source gives
    them exact targets - breadth-first search from the central state up to `depth` knows the true distance of every
    state within that depth, and states are sampled from what it found.

    For a Q-model (`n_outputs` equal to the number of generators), states are sampled so that all their children are
    within `depth` as well, and every output gets an exact target - nothing is masked.

    A few per cent of anchors in the training data is enough: it is what keeps predictions near the central state
    honest, while a large share of them makes the model good there and worse everywhere else, which is where beam
    search spends most of its time.

    Example:

    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.train import BfsAnchors
    >>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    >>> data = BfsAnchors(graph, depth=2, size=5, n_outputs=3).generate()
    >>> tuple(data.targets.shape)
    (5, 3)
    """

    def __init__(
        self,
        graph: "CayleyGraph",
        depth: int,
        size: Optional[int] = None,
        n_outputs: int = 1,
        max_table_states: int = 10**7,
    ):
        """Initializes BfsAnchors and runs the breadth-first search (once - :meth:`generate` only samples from it).

        :param graph: Graph to search in.
        :param depth: How many layers of breadth-first search to compute. Must be at least 1. Deeper means more states
            with exact distances, and quickly more memory.
        :param size: How many states :meth:`generate` returns, sampled with replacement. Defaults to None, meaning all
            states with exact distances.
        :param n_outputs: Number of outputs of the model to train - 1 for a model predicting the distance of a state,
            or the number of generators for a Q-model predicting distances of all children of a state.
        :param max_table_states: Safety limit on the number of states with exact distances. The search stops when it is
            exceeded and the constructor fails, instead of running the machine out of memory.
        """
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}.")
        if size is not None:
            _check_positive("size", size)
        _check_positive("max_table_states", max_table_states)
        n_generators = graph.definition.n_generators
        if n_outputs not in (1, n_generators):
            raise ValueError(
                f"n_outputs must be either 1 or the number of generators of this graph ({n_generators}), got "
                f"{n_outputs}."
            )
        self.graph = graph
        self.depth = int(depth)
        self.size = size
        self.n_outputs = int(n_outputs)
        self.max_table_states = int(max_table_states)

        n_states_found = 1
        exceeded_limit = False

        def stop_condition(layer: torch.Tensor, _hashes: torch.Tensor) -> bool:
            nonlocal n_states_found, exceeded_limit
            n_states_found += int(layer.shape[0])
            exceeded_limit = n_states_found > self.max_table_states
            return exceeded_limit

        bfs_result = graph.bfs(max_diameter=self.depth, max_layer_size_to_store=None, stop_condition=stop_condition)
        if exceeded_limit:
            raise ValueError(
                f"Breadth-first search up to depth {self.depth} was stopped after it found {n_states_found} states, "
                f"which is more than max_table_states={self.max_table_states}. Use a smaller depth, or a larger limit "
                "if this many states fit in memory."
            )

        layer_ids = sorted(bfs_result.layers)
        states = torch.vstack([bfs_result.layers[i] for i in layer_ids])
        distances = torch.cat(
            [
                torch.full((int(bfs_result.layers[i].shape[0]),), i, dtype=torch.int64, device=states.device)
                for i in layer_ids
            ]
        )
        hashes = graph.hasher.make_hashes(graph.encode_states(states))
        order = torch.argsort(hashes)
        self._table_hashes = hashes[order]
        self._table_distances = distances[order]

        # A state can only be an anchor for a Q-model if all of its children have exact distances too. That holds for
        # states of the last computed layer only if the search explored the whole graph.
        if self.n_outputs == 1 or bfs_result.bfs_completed:
            is_anchor = torch.ones_like(distances, dtype=torch.bool)
        else:
            is_anchor = distances < layer_ids[-1]
        self._states = states[is_anchor]
        self._distances = distances[is_anchor].to(torch.float32)

    @property
    def n_states_with_exact_distance(self) -> int:
        """Number of states whose exact distance is known (found by the breadth-first search)."""
        return int(self._table_hashes.shape[0])

    def generate(self) -> TrainingData:
        """Samples states with exact distances.

        :return: States with their exact distances (for a Q-model - with exact distances of all their children).
        """
        graph = self.graph
        index = _sample_index(int(self._states.shape[0]), self.size, self._states.device)
        states = self._states[index]
        if self.n_outputs == 1:
            return TrainingData(states=states, targets=self._distances[index])
        n_states = int(states.shape[0])
        n_generators = graph.definition.n_generators
        children = graph.get_neighbors(graph.encode_states(states))
        distances, found = self._lookup(graph.hasher.make_hashes(children))
        if not bool(found.all()):
            raise RuntimeError("Some children of anchor states have no exact distance, which should not happen.")
        # get_neighbors returns children of all states for the first generator, then for the second one, and so on.
        targets = distances.reshape((n_generators, n_states)).transpose(0, 1).to(torch.float32).contiguous()
        return TrainingData(states=states, targets=targets)

    def _lookup(self, hashes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Looks up exact distances of states with given hashes.

        :param hashes: Hashes of states to look up.
        :return: Pair of tensors (distances, found), where ``found[i]`` says whether the distance of the state with
            hash ``hashes[i]`` is known (if it is not, ``distances[i]`` is meaningless).
        """
        index = torch.searchsorted(self._table_hashes, hashes).clamp(max=self.n_states_with_exact_distance - 1)
        return self._table_distances[index], self._table_hashes[index] == hashes


class PathDataSource(DataSource):
    """Training data from paths to the central state, e.g. from paths found by beam search.

    A path of length ``L`` gives a target for every state on it in hindsight: the state visited after ``p`` moves is
    ``L-p`` moves away from the central state. These targets are upper bounds on the true distance (the path does not
    have to be optimal), which is what `weight` is for - it makes the loss trust them less than exact targets.

    This is the way to train on states that nothing else reaches: solutions of hard instances found by a previous run
    of beam search, or published solutions of a competition (see :meth:`from_tsv`).

    A path is a :class:`cayleypy.CayleyPath`, so a solution found by beam search becomes training data as::

        result = graph.beam_search(start_state=state, predictor=predictor, return_path=True)
        source = PathDataSource(graph, [CayleyPath(state, result.path, graph.definition)])
    """

    def __init__(
        self,
        graph: "CayleyGraph",
        paths: Sequence[CayleyPath],
        n_outputs: int = 1,
        weight: float = 1.0,
        size: Optional[int] = None,
    ):
        """Initializes PathDataSource and computes the targets (once - :meth:`generate` only samples from them).

        :param graph: Graph the paths are in.
        :param paths: Paths to the central state. Every path must end at the central state, otherwise the number of
            remaining moves is not an estimate of the distance.
        :param n_outputs: Number of outputs of the model to train - 1 for a model predicting the distance of a state,
            or the number of generators for a Q-model, in which case the target is for the output of the move that the
            path takes and all the other outputs are masked out.
        :param weight: Importance of these targets for the loss, relative to targets of other sources (which is 1 by
            default). Less than 1 means "these are upper bounds, do not trust them as much as exact distances".
        :param size: How many states :meth:`generate` returns, sampled with replacement. Defaults to None, meaning all
            states on all paths.
        """
        n_generators = graph.definition.n_generators
        if n_outputs not in (1, n_generators):
            raise ValueError(
                f"n_outputs must be either 1 or the number of generators of this graph ({n_generators}), got "
                f"{n_outputs}."
            )
        _check_positive("weight", weight)
        if size is not None:
            _check_positive("size", size)
        if len(paths) == 0:
            raise ValueError("At least one path is needed.")
        self.graph = graph
        self.n_outputs = int(n_outputs)
        self.weight = float(weight)
        self.size = size
        self._data = TrainingData.concat([self._data_for_path(path) for path in paths])

    @staticmethod
    def from_tsv(
        path: PathType,
        graph: "CayleyGraph",
        n_outputs: int = 1,
        weight: float = 1.0,
        size: Optional[int] = None,
        puzzle_id: Optional[str] = None,
        column: str = _DEFAULT_SOLUTION_COLUMN,
        delimiter: str = ".",
    ) -> "PathDataSource":
        """Reads solutions from a tab-separated file, in the format published in the "cayleypy-beam-results" repository.

        The file must have a header row and a column of solutions, where a solution is a list of names of generators
        leading from the state that was solved to the central state (the format of
        :meth:`cayleypy.CayleyGraphDef.path_to_string`). All the other columns are ignored, so the wide tables of that
        repository can be read as they are.

        The state that was solved does not have to be in the file: every solution is replayed backwards from the
        central state, which needs the generators of `graph` to be inverse closed.

        :param path: Path to the file to read.
        :param graph: Graph the solutions are for. Names of its generators must be the names used in the file.
        :param n_outputs: See :class:`PathDataSource`.
        :param weight: See :class:`PathDataSource`.
        :param size: See :class:`PathDataSource`.
        :param puzzle_id: If set, only rows with this value in the "puzzle_id" column are read.
        :param column: Name of the column with solutions.
        :param delimiter: Delimiter between names of generators inside that column.
        :return: The data source.
        """
        inverse_map = graph.definition.generators_inverse_map
        if inverse_map is None:
            raise ValueError(
                "Generators of this graph are not inverse closed, so solutions cannot be replayed backwards from the "
                "central state. Consider reading them for the graph returned by "
                "CayleyGraph.with_inverted_generators()."
            )
        generator_ids = {name: i for i, name in enumerate(graph.definition.generator_names)}
        paths = []
        with open(path, "r", encoding="utf-8", newline="") as file:
            for row_id, row in enumerate(csv.DictReader(file, delimiter="\t")):
                if puzzle_id is not None and row.get("puzzle_id", "").strip(_CELL_PREFIXES) != puzzle_id:
                    continue
                if column not in row:
                    raise ValueError(f'File has no column "{column}", its columns are: {sorted(row)}.')
                solution = (row[column] or "").strip(_CELL_PREFIXES)
                if solution == "":
                    continue
                edges = []
                for name in solution.split(delimiter):
                    if name not in generator_ids:
                        raise ValueError(
                            f'Row {row_id} of the file uses move "{name}", which is not a generator of this graph. Its '
                            f"generators are: {graph.definition.generator_names}."
                        )
                    edges.append(generator_ids[name])
                paths.append(_path_ending_at_central_state(graph, edges, inverse_map))
        if len(paths) == 0:
            raise ValueError(f"No solutions were read from {path}.")
        return PathDataSource(graph, paths, n_outputs=n_outputs, weight=weight, size=size)

    def generate(self) -> TrainingData:
        """Samples states on the paths with their hindsight targets.

        :return: States with the number of moves that remained to the central state as their target distance.
        """
        index = _sample_index(len(self._data), self.size, self._data.states.device)
        return self._data.select(index)

    def _data_for_path(self, path: CayleyPath) -> TrainingData:
        """Computes states on one path with their hindsight targets."""
        graph = self.graph
        n_generators = graph.definition.n_generators
        for gen_id in path.edges:
            if not 0 <= gen_id < n_generators:
                raise ValueError(f"Path uses generator {gen_id}, but this graph has {n_generators} generators.")
        states = _states_on_path(graph, path)
        if not torch.equal(states[-1].reshape(-1), graph.central_state.reshape(-1)):
            raise ValueError("Path does not end at the central state, so it says nothing about distances.")
        path_length = len(path.edges)
        remaining_moves = torch.arange(path_length, -1, -1, device=states.device, dtype=torch.float32)
        if self.n_outputs == 1:
            weights = torch.full_like(remaining_moves, self.weight)
            return TrainingData(states=states, targets=remaining_moves, weights=weights)
        # A move of the path leads to the next state on it, so only the output of that move is labeled. The last state
        # is the central state, from which the path takes no move, so it is not included.
        moves = torch.tensor(path.edges, device=states.device, dtype=torch.int64).unsqueeze(1)
        outputs = torch.arange(n_generators, device=states.device).unsqueeze(0)
        mask = outputs == moves
        targets = remaining_moves[1:].unsqueeze(1).expand(path_length, n_generators)
        return TrainingData(
            states=states[:-1],
            targets=targets.contiguous(),
            mask=mask,
            weights=torch.full_like(targets, self.weight),
        )


def _states_on_path(graph: "CayleyGraph", path: CayleyPath) -> torch.Tensor:
    """Returns all states on a path (in decoded representation), including its first and last states."""
    current = graph.encode_states(path.start_state)
    encoded = [current]
    for gen_id in path.edges:
        next_state = torch.zeros_like(current)
        graph.apply_generator_batched(gen_id, current, next_state)
        encoded.append(next_state)
        current = next_state
    return graph.decode_states(torch.vstack(encoded))


def _path_ending_at_central_state(graph: "CayleyGraph", edges: list[int], inverse_map: list[int]) -> CayleyPath:
    """Restores the path with given edges that ends at the central state, by replaying it backwards from that state."""
    state = graph.encode_states(graph.central_state)
    for gen_id in reversed(edges):
        previous_state = torch.zeros_like(state)
        graph.apply_generator_batched(inverse_map[gen_id], state, previous_state)
        state = previous_state
    return CayleyPath(graph.decode_states(state).reshape(-1), edges, graph.definition)


class MixtureDataSource(DataSource):
    """Several data sources mixed in given proportions.

    Every source generates its data, and then this source takes a random part of every piece, so that the proportions
    are the requested ones. Sizes are chosen as large as they can be without repeating states, i.e. the total number of
    states is ``min(len(data of source i) / fractions[i])``: if one source generates less data than its share, the
    result is smaller, not filled with copies.

    Example:

    >>> from cayleypy import CayleyGraph, PermutationGroups
    >>> from cayleypy.train import BfsAnchors, MixtureDataSource, RandomWalksSource
    >>> graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    >>> walks = RandomWalksSource(graph, n_walks=8, rw_length=5)
    >>> anchors = BfsAnchors(graph, depth=2, size=4)
    >>> len(MixtureDataSource([walks, anchors], [0.9, 0.1]).generate())
    40
    """

    def __init__(self, sources: Sequence[DataSource], fractions: Optional[Sequence[float]] = None):
        """Initializes MixtureDataSource.

        :param sources: Sources to mix. Must be non-empty.
        :param fractions: Share of every source in the generated data. Must be positive and sum to 1. Defaults to
            equal shares.
        """
        if len(sources) == 0:
            raise ValueError("At least one data source is needed.")
        if fractions is None:
            fractions = [1.0 / len(sources)] * len(sources)
        if len(fractions) != len(sources):
            raise ValueError(f"There are {len(sources)} sources, but {len(fractions)} fractions for them.")
        for fraction in fractions:
            _check_positive("fraction", fraction)
        total = sum(fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1, but they sum to {total}.")
        self.sources = list(sources)
        self.fractions = [float(fraction) for fraction in fractions]

    def generate(self) -> TrainingData:
        """Generates data from all sources and mixes them in the configured proportions.

        :return: Data from all sources, in one piece.
        """
        parts = [source.generate() for source in self.sources]
        n_states = min(len(part) / fraction for part, fraction in zip(parts, self.fractions))
        pieces = []
        for part, fraction in zip(parts, self.fractions):
            # Every source that generated something contributes at least one state, so that a source is never silently
            # dropped because its share rounds down to zero.
            size = min(len(part), max(1, int(round(fraction * n_states)))) if len(part) > 0 else 0
            pieces.append(part.select(torch.randperm(len(part), device=part.states.device)[:size]))
        return TrainingData.concat(pieces)
