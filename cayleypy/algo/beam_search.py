"""Beam search algorithm for Cayley graphs."""

import time
from typing import NamedTuple, TYPE_CHECKING, Optional

import numpy as np
import torch

from .beam_search_result import BeamSearchResult
from .bfs_result import BfsResult
from ..cayley_graph_def import AnyStateType
from ..predictor import Predictor
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph
    from ..lower_bound import LowerBound
    from ..symmetries import SymmetryGroup


class _ExpandedLayer(NamedTuple):
    """States obtained by applying all generators to states of the current layer, with duplicates removed.

    All fields are parallel arrays, indexed by position of a state on the next layer.
    """

    states: torch.Tensor
    """Unique states of the next layer (in internal representation)."""

    hashes: torch.Tensor
    """Hashes of `states`."""

    moves: torch.Tensor
    """Id of the generator that produced each state.

    This is the provenance of a state, which is lost when the next layer is deduplicated as a flat set of states.
    """

    source_index: torch.Tensor
    """Index of each state in the output of `CayleyGraph.get_neighbors`, used to look up score of that state."""


def _unique_index(hashes: torch.Tensor) -> torch.Tensor:
    """Returns indexes of the first state with each distinct hash, sorted by hash.

    :param hashes: Hashes of states, one per state.
    :return: Indexes of states to keep, so that no two of them have equal hashes.
    """
    hashes_sorted, idx = torch.sort(hashes, stable=True)
    # Compute mask of first occurrences for each unique value.
    mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=hashes_sorted.device)
    if hashes_sorted.size(0) > 1:
        mask[1:] = hashes_sorted[1:] != hashes_sorted[:-1]
    return idx[mask]


def _expand_layer(
    graph: "CayleyGraph", states: torch.Tensor, banned_moves: Optional[torch.Tensor] = None
) -> _ExpandedLayer:
    """Applies all generators to `states` and removes duplicates, remembering where each state came from.

    This is a variant of `CayleyGraph.get_unique_states` that additionally keeps the link between a state on the next
    layer and the generator that produced it. Deduplication is the same: states are sorted by hash, and the first
    state with each hash is kept.

    :param graph: The Cayley graph.
    :param states: States of the current layer (in internal representation).
    :param banned_moves: Id of one generator per state, which must not be applied to that state. Banned moves are
        excluded before deduplication, so a state that is also produced by some allowed move is still returned.
        Defaults to None, which means all generators are applied to all states.
    :return: States of the next layer, with provenance of each state.
    """
    n_states = int(states.shape[0])
    neighbors = graph.get_neighbors(states)
    hashes = graph.hasher.make_hashes(neighbors)
    # `get_neighbors` writes neighbors in generator-major order: rows [i*n_states, (i+1)*n_states) are obtained by
    # applying generator i, so the id of that generator is the row index divided by the number of states.
    if banned_moves is None:
        unique_idx = _unique_index(hashes)
    else:
        allowed = torch.ones((graph.definition.n_generators, n_states), dtype=torch.bool, device=states.device)
        allowed[banned_moves, torch.arange(n_states, device=states.device)] = False
        allowed_idx = allowed.reshape(-1).nonzero().reshape(-1)
        unique_idx = allowed_idx[_unique_index(hashes[allowed_idx])]
    moves = torch.div(unique_idx, n_states, rounding_mode="floor")
    return _ExpandedLayer(neighbors[unique_idx], hashes[unique_idx], moves, unique_idx)


def _inverse_generators(graph: "CayleyGraph") -> torch.Tensor:
    """Returns id of the inverse generator for each generator, as a tensor indexed by generator id.

    :param graph: The Cayley graph.
    :return: Tensor of shape `[n_generators]`, where element i is id of the generator inverse to generator i.
    """
    inverse_map = graph.definition.generators_inverse_map
    if inverse_map is None:
        raise ValueError(
            "non_backtracking requires inverse-closed generators (for every generator, its inverse must also be a "
            "generator), which is not the case for this graph."
        )
    return torch.tensor(inverse_map, dtype=torch.int64, device=graph.device)


def _check_symmetries_match_graph(graph: "CayleyGraph", symmetry_group: "SymmetryGroup") -> None:
    """Checks that `symmetry_group` was created for `graph` (and not for some other graph)."""
    graph_def = graph.definition
    symmetries_def = symmetry_group.graph_def
    if (
        graph_def.generators_permutations != symmetries_def.generators_permutations
        or graph_def.central_state != symmetries_def.central_state
    ):
        raise ValueError("Symmetry group was created for a different graph than the one being searched.")


def _dedup_by_canonical_form(graph: "CayleyGraph", states: torch.Tensor, symmetry_group: "SymmetryGroup"):
    """Returns indexes of states to keep, so that exactly one state from each orbit under symmetries remains.

    Which state of an orbit is kept is unspecified (it is the first one in the order of hashes of canonical forms).

    :param graph: The Cayley graph.
    :param states: States to deduplicate (in internal representation).
    :param symmetry_group: Symmetries of the graph. States equivalent under these are considered duplicates.
    :return: Indexes of states to keep.
    """
    canonical_states = symmetry_group.canonical(graph.decode_states(states))
    return _unique_index(graph.hasher.make_hashes(graph.encode_states(canonical_states)))


def _states_within_budget(
    graph: "CayleyGraph", states: torch.Tensor, lower_bound: "LowerBound", budget: int
) -> torch.Tensor:
    """Returns mask of states from which the central state can still be reached in `budget` steps or less.

    :param graph: The Cayley graph.
    :param states: States to filter (in internal representation).
    :param lower_bound: Admissible lower bound on distance from a state to the central state.
    :param budget: How many steps are left, i.e. the maximal allowed distance from a state to the central state.
    :return: Boolean mask of states to keep.
    """
    bounds = lower_bound.lb(graph.decode_states(states))
    if tuple(bounds.shape) != (int(states.shape[0]),):
        raise ValueError(
            f"lower_bound.lb returned bounds of shape {tuple(bounds.shape)}, but shape {(int(states.shape[0]),)} "
            "was expected."
        )
    return bounds <= budget


def _filter_layer(
    keep: torch.Tensor, states: torch.Tensor, hashes: torch.Tensor, expanded: Optional[_ExpandedLayer]
) -> tuple[torch.Tensor, torch.Tensor, Optional[_ExpandedLayer]]:
    """Keeps only some states of a layer, keeping their provenance matched with them.

    :param keep: Indexes of states to keep, or a boolean mask of states to keep.
    :param states: States of the layer (in internal representation).
    :param hashes: Hashes of `states`.
    :param expanded: Provenance of `states`, or None if it is not tracked.
    :return: The remaining states, their hashes and their provenance.
    """
    states, hashes = states[keep, :], hashes[keep]
    if expanded is not None:
        expanded = _ExpandedLayer(states, hashes, expanded.moves[keep], expanded.source_index[keep])
    return states, hashes, expanded


def _score_children(graph: "CayleyGraph", predictor: Predictor, states: torch.Tensor) -> torch.Tensor:
    """Estimates scores of all children of `states`, calling the predictor once per state (not once per child).

    :param graph: The Cayley graph.
    :param predictor: A heuristic that estimates scores for states.
    :param states: States of the current layer (in internal representation).
    :return: Scores of all children, flattened in the same order in which `CayleyGraph.get_neighbors` returns them.
    """
    n_states = int(states.shape[0])
    n_generators = graph.definition.n_generators
    scores = predictor.score_children(graph.decode_states(states))
    if tuple(scores.shape) != (n_states, n_generators):
        raise ValueError(
            f"score_children returned scores of shape {tuple(scores.shape)}, but shape {(n_states, n_generators)} "
            "was expected. Model used with child scoring must have exactly one output per generator."
        )
    # `score_children` returns scores state-major, while `get_neighbors` returns children generator-major.
    return scores.transpose(0, 1).reshape(-1)


class BeamSearchAlgorithm:
    """Beam search algorithm for finding paths in Cayley graphs.

    This class implements the beam search algorithm to find paths from a given start state
    to the central state of a Cayley graph. It can use various heuristics (predictors) to
    guide the search and supports meet-in-the-middle optimization.
    """

    def __init__(self, graph: "CayleyGraph"):
        """Initialize the beam search algorithm.

        :param graph: The Cayley graph to search on.
        """
        self.graph = graph

    def search(
        self,
        *,
        start_state: AnyStateType,
        destination_state: Optional[AnyStateType] = None,
        beam_mode: str = "simple",
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        history_depth: int = 0,
        return_path: bool = False,
        bfs_result_for_mitm: Optional[BfsResult] = None,
        use_child_scores: bool = False,
        canonical_dedup: Optional["SymmetryGroup"] = None,
        non_backtracking: bool = False,
        lower_bound: Optional["LowerBound"] = None,
        prune_above: Optional[int] = None,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to destination state using Beam Search algorithm.

        The following beam search modes are supported:

          * "simple" - classic beam search algorithm that finds paths from start state to central state.
            Uses meet-in-the-middle optimization if `bfs_result_for_mitm` is provided.
            Supports path restoration if `return_path=True`.
          * "advanced" - enhanced beam search with non-backtracking capabilities.
            Supports configurable history depth to avoid revisiting states.
            Uses PyTorch for efficient batch processing.

        :param start_state: State from which to start search.
        :param destination_state: Target state to find. Defaults to central state for "simple" mode.
        :param beam_mode: Type of beam search (see above). Defaults to "simple".
        :param predictor: A heuristic that estimates scores for states (lower score = closer to destination).
          Defaults to Hamming distance heuristic.
        :param beam_width: Width of the beam (how many "best" states we consider at each step).
        :param max_steps: Maximum number of search steps/iterations before giving up.
        :param history_depth: For "advanced" mode, how many previous levels to remember and ban from revisiting.
        :param return_path: For "simple" mode, whether to return path (consumes much more memory if True).
        :param bfs_result_for_mitm: For "simple" mode, BfsResult with pre-computed neighborhood of central state
            for meet-in-the-middle optimization. Defaults to None.
        :param use_child_scores: For "simple" mode, whether to score states using `Predictor.score_children`
            (see :meth:`search_simple`). Defaults to False.
        :param canonical_dedup: For "simple" mode, symmetries by which to deduplicate the beam
            (see :meth:`search_simple`). Defaults to None, which means no deduplication by symmetries.
        :param non_backtracking: For "simple" mode, whether to ban moves undoing the previous move
            (see :meth:`search_simple`). In "advanced" mode, use `history_depth` instead. Defaults to False.
        :param lower_bound: For "simple" mode, admissible lower bound on distance to the central state, used together
            with `prune_above` (see :meth:`search_simple`). Defaults to None, which means no pruning.
        :param prune_above: For "simple" mode, maximal length of a path we are interested in
            (see :meth:`search_simple`). Defaults to None, which means no pruning.
        :param verbose: Verbosity level (0=quiet, 1=basic, 10=detailed, 100=profiling).
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        if beam_mode == "simple":
            return self.search_simple(
                start_state=start_state,
                predictor=predictor,
                beam_width=beam_width,
                max_steps=max_steps,
                return_path=return_path,
                bfs_result_for_mitm=bfs_result_for_mitm,
                use_child_scores=use_child_scores,
                canonical_dedup=canonical_dedup,
                non_backtracking=non_backtracking,
                lower_bound=lower_bound,
                prune_above=prune_above,
            )
        elif beam_mode == "advanced":
            if use_child_scores:
                raise ValueError("use_child_scores is supported only in 'simple' beam mode.")
            if canonical_dedup is not None:
                raise ValueError("canonical_dedup is supported only in 'simple' beam mode.")
            if non_backtracking:
                raise ValueError("non_backtracking is supported only in 'simple' beam mode, use history_depth here.")
            if lower_bound is not None or prune_above is not None:
                raise ValueError("lower_bound and prune_above are supported only in 'simple' beam mode.")
            return self.search_advanced(
                start_state=start_state,
                destination_state=destination_state,
                beam_width=beam_width,
                max_steps=max_steps,
                history_depth=history_depth,
                predictor=predictor,
                verbose=verbose,
            )
        else:
            raise ValueError("Unknown beam_mode:", beam_mode)

    def search_simple(
        self,
        *,
        start_state: AnyStateType,
        predictor: Optional[Predictor] = None,
        beam_width=1000,
        max_steps=1000,
        return_path=False,
        bfs_result_for_mitm: Optional[BfsResult] = None,
        use_child_scores: bool = False,
        canonical_dedup: Optional["SymmetryGroup"] = None,
        non_backtracking: bool = False,
        lower_bound: Optional["LowerBound"] = None,
        prune_above: Optional[int] = None,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to central state using simple Beam Search algorithm.

        :param start_state: State from which to start search.
        :param predictor: A heuristic that estimates scores for states (lower score = closer to center).
          Defaults to Hamming distance heuristic.
        :param beam_width: Width of the beam (how many "best" states we consider at each step).
        :param max_steps: Maximum number of iterations before giving up.
        :param return_path: Whether to return path (consumes much more memory if True).
        :param bfs_result_for_mitm: BfsResult with pre-computed neighborhood of central state to compute for
            meet-in-the-middle modification of Beam Search. Beam search will terminate when any of states in that
            neighborhood is encountered. Defaults to None, which means no meet-in-the-middle (i.e. only search for the
            central state).
        :param use_child_scores: Whether to score states on the next layer with `Predictor.score_children` applied to
            states on the current layer, instead of applying the predictor to the states on the next layer themselves.
            Scores are the same, but the predictor is called once per state of the current layer instead of once per
            state of the next layer, which is much cheaper for models having one output per generator (Q-models).
            Defaults to False, which means the predictor is applied to states on the next layer.
        :param canonical_dedup: Symmetries of the graph (created for the same graph) by which to deduplicate states
            on the next layer, keeping only one state from each orbit. States equivalent under a symmetry have equal
            distances to the central state, so exploring only one of them makes the beam cover more distinct states
            (in effect, this multiplies the beam width by up to the number of symmetries). Deduplication happens after
            the check whether the central state is reached, so the path is never lost. Defaults to None, which means
            states are deduplicated only by equality, as usual.
        :param non_backtracking: Whether to ban the move inverse to the move by which a state was reached. Such a move
            leads back to a state on the previous layer, so banning it makes the beam cover more distinct states. This
            is similar to `history_depth=1` in "advanced" mode, but it needs no memory to store hashes of the previous
            layer. A state is banned only along the recorded move: if it is also produced by another move (from
            another state of the beam), it stays. This pays off when the beam is narrow enough to be truncated; for a
            beam wide enough to hold the whole layer, banning only removes states from it. Requires the generators to
            be inverse-closed. Defaults to False.
        :param lower_bound: Admissible lower bound on distance from a state to the central state
            (see :class:`cayleypy.LowerBound`), used to drop states that cannot be on a path of length at most
            `prune_above`. Dropping them makes room in the beam for states that still can. Must be specified together
            with `prune_above`. Defaults to None, which means no pruning.
        :param prune_above: Maximal length of a path we are interested in, usually a path length from a previous run
            that we want to improve on. A state reached in `k` steps is dropped when ``k + lower_bound(state)`` is
            greater than this. Paths longer than this may still be found (pruning only guarantees that no path of
            length at most `prune_above` is lost). Must be specified together with `lower_bound`. Defaults to None,
            which means no pruning.
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        graph = self.graph
        if predictor is None:
            predictor = Predictor(graph, "hamming")
        if canonical_dedup is not None:
            _check_symmetries_match_graph(graph, canonical_dedup)
        if (lower_bound is None) != (prune_above is None):
            raise ValueError(
                "lower_bound and prune_above must be specified together: without a lower bound there is nothing to "
                "compare with prune_above, and without prune_above the lower bound has no effect."
            )
        if prune_above is not None and prune_above < 0:
            raise ValueError(f"prune_above must be non-negative, got {prune_above}.")
        inverse_generators = _inverse_generators(graph) if non_backtracking else None

        start_states = graph.encode_states(start_state)
        layer1, layer1_hashes = graph.get_unique_states(start_states)
        all_layers_hashes = [layer1_hashes]
        debug_scores = {}  # type: dict[int, float]

        if graph.central_state_hash[0] == layer1_hashes[0]:
            # Start state is the central state.
            return BeamSearchResult(True, 0, [], debug_scores, graph.definition)

        bfs_layers_hashes = [graph.central_state_hash]
        if bfs_result_for_mitm is not None:
            assert bfs_result_for_mitm.graph == graph.definition
            bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Checks if any of `hashes` are in neighborhood of the central state.
        # Returns the number of the first layer where intersection was found, or -1 if not found.
        def _check_path_found(hashes):
            for j, layer in enumerate(bfs_layers_hashes):
                if torch.any(isin_via_searchsorted(layer, hashes)):
                    return j
            return -1

        def _restore_path(found_layer_id: int) -> Optional[list[int]]:
            if not return_path:
                return None
            if found_layer_id == 0:
                return graph.restore_path(all_layers_hashes, graph.central_state)
            assert bfs_result_for_mitm is not None
            mask = isin_via_searchsorted(layer2_hashes, bfs_layers_hashes[found_layer_id])
            assert torch.any(mask), "No intersection in Meet-in-the-middle."
            middle_state = graph.decode_states(layer2[mask.nonzero()[0].item()].reshape((1, -1)))
            path1 = graph.restore_path(all_layers_hashes, middle_state)
            path2 = graph.find_path_from(middle_state, bfs_result_for_mitm)
            assert path2 is not None
            return path1 + path2

        # Moves banned on the current layer (one per state), None if nothing is banned.
        banned_moves: Optional[torch.Tensor] = None
        # When children are scored through their parents, or when moves undoing the previous move are banned, we need to
        # know which (state, generator) pair produced each state of the next layer.
        need_provenance = use_child_scores or non_backtracking

        for i in range(max_steps):
            # Create states on the next layer.
            expanded = _expand_layer(graph, layer1, banned_moves) if need_provenance else None
            if expanded is not None:
                layer2, layer2_hashes = expanded.states, expanded.hashes
            else:
                layer2, layer2_hashes = graph.get_unique_states(graph.get_neighbors(layer1))

            if layer2.shape[0] == 0:
                # Can happen only with non_backtracking: the only move from every state of the beam was banned.
                return BeamSearchResult(False, 0, None, debug_scores, graph.definition)

            bfs_layer_id = _check_path_found(layer2_hashes)
            if bfs_layer_id != -1:
                # Path found.
                path = _restore_path(bfs_layer_id)
                return BeamSearchResult(True, i + bfs_layer_id + 1, path, debug_scores, graph.definition)

            if canonical_dedup is not None:
                # Keep only one state from each orbit under symmetries. Every kept state is still reachable from the
                # start state in `i+1` steps through kept states of the previous layers, so path restoration and the
                # reported path length remain correct.
                keep = _dedup_by_canonical_form(graph, layer2, canonical_dedup)
                layer2, layer2_hashes, expanded = _filter_layer(keep, layer2, layer2_hashes, expanded)

            if lower_bound is not None and prune_above is not None:
                # Drop states from which the central state cannot be reached within the remaining budget of steps.
                # States of this layer were reached in `i+1` steps, so `prune_above-(i+1)` steps are left.
                keep = _states_within_budget(graph, layer2, lower_bound, prune_above - (i + 1))
                layer2, layer2_hashes, expanded = _filter_layer(keep, layer2, layer2_hashes, expanded)
                if layer2.shape[0] == 0:
                    # No state can be on a path of length at most `prune_above`, so there is no such path.
                    return BeamSearchResult(False, 0, None, debug_scores, graph.definition)

            layer2_moves = expanded.moves if expanded is not None else None

            # Pick `beam_width` states with lowest scores.
            if len(layer2) >= beam_width:
                if expanded is not None and use_child_scores:
                    scores = _score_children(graph, predictor, layer1)[expanded.source_index]
                else:
                    scores = predictor(graph.decode_states(layer2))
                idx = torch.argsort(scores)[:beam_width]
                layer2 = layer2[idx, :]
                layer2_hashes = layer2_hashes[idx]
                if layer2_moves is not None:
                    layer2_moves = layer2_moves[idx]
                best_score = float(scores[idx[0]].detach())
                debug_scores[i] = best_score
                if graph.verbose >= 2:
                    print(f"Iteration {i}, best score {best_score}.")

            layer1 = layer2
            layer1_hashes = layer2_hashes
            if non_backtracking:
                assert inverse_generators is not None and layer2_moves is not None
                banned_moves = inverse_generators[layer2_moves]
            if return_path:
                all_layers_hashes.append(layer1_hashes)

        # Path not found.
        return BeamSearchResult(False, 0, None, debug_scores, graph.definition)

    def search_advanced(
        self,
        start_state: AnyStateType,
        destination_state: Optional[AnyStateType] = None,
        *,
        beam_width: int = 1000,
        max_steps: int = 1000,
        history_depth: int = 0,
        predictor: Optional[Predictor] = None,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Advanced beam search using PyTorch with non-backtracking capabilities.

        This method implements an improved beam search algorithm that supports:
        - Non-backtracking constraints (avoiding revisiting states)
        - Batch processing for efficiency
        - Configurable history depth for state banning

        :param start_state: State from which to start search.
        :param destination_state: Target state to find. Defaults to central state.
        :param beam_width: Width of the beam (how many best states to consider).
        :param max_steps: Maximum number of search steps.
        :param history_depth: How many previous levels to remember and ban from revisiting.
        :param predictor: Predictor object for scoring states. If None, uses Hamming distance.
        :param batch_size: Batch size for model predictions.
        :param verbose: Verbosity level (0=quiet, 1=basic, 10=detailed, 100=profiling).
        :return: BeamSearchResult with search results.
        """
        graph = self.graph

        # Use central state as destination if not specified.
        if destination_state is None:
            destination_state = graph.central_state

        # Encode states.
        start_states_encoded = graph.encode_states(start_state)
        destination_states_encoded = graph.encode_states(destination_state)

        # Initialize predictor if not provided.
        if predictor is None:
            predictor = Predictor(graph, "hamming")

        # Check if start state is already the destination.
        if torch.all(start_states_encoded == destination_states_encoded):
            return BeamSearchResult(True, 0, [], {}, graph.definition)

        # Initialize beam states.
        array_beam_states = start_states_encoded.view(1, -1).clone()

        # Initialize hash storage for non-backtracking.
        if history_depth > 0:
            initial_hashes = graph.hasher.make_hashes(start_states_encoded)
            vec_hashes_current = initial_hashes.expand(
                beam_width * graph.definition.n_generators, history_depth
            ).clone()
            i_cyclic_index_for_hash_storage = 0

        debug_scores: dict[int, float] = {}
        t0 = time.time()

        for i_step in range(1, max_steps + 1):
            t_moves = t_hash = t_isin = t_unique_els = 0.0
            t_full_step = time.time()

            # Create new states by applying all generators.
            t1 = time.time()
            array_new_states = graph.get_neighbors(array_beam_states)
            # Ensure it's 2D: (n_states, state_size).
            if array_new_states.dim() == 1:
                array_new_states = array_new_states.unsqueeze(0)
            elif array_new_states.dim() > 2:
                array_new_states = array_new_states.flatten(end_dim=1)
            t_moves += time.time() - t1

            # Take only unique states.
            t1 = time.time()
            array_new_states, _ = graph.get_unique_states(array_new_states)
            t_unique_els += time.time() - t1

            # Check if destination state is found.
            vec_tmp = torch.all(array_new_states == destination_states_encoded, dim=1)
            flag_found_destination = torch.any(vec_tmp).item()
            if flag_found_destination:
                if verbose >= 1:
                    print(f"Found destination state at step {i_step}, ways: {vec_tmp.sum().item()}.")
                return BeamSearchResult(True, i_step, None, debug_scores, graph.definition)

            # Non-backtracking: forbid visiting states visited before.
            if history_depth > 0:
                t1 = time.time()
                vec_hashes_new = graph.hasher.make_hashes(array_new_states)
                t_hash += time.time() - t1

                t1 = time.time()
                mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
                t_isin += time.time() - t1
                mask_new_sum = mask_new.sum().item()

                if mask_new_sum > 0:
                    array_new_states = array_new_states[mask_new, :]
                else:
                    if verbose >= 1:
                        print(f"Cannot find new states at step {i_step}.")
                    return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

                # Update hash storage.
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth
                i_tmp = len(vec_hashes_new)
                vec_hashes_current[:i_tmp, i_cyclic_index_for_hash_storage] = vec_hashes_new

            # Estimate states and select top beam_width ones.
            t_predict = time.time()
            if array_new_states.shape[0] > beam_width:
                # Score states using predictor.
                scores = predictor(graph.decode_states(array_new_states))

                # Select best states.
                if isinstance(scores, torch.Tensor):
                    idx = torch.argsort(scores)[:beam_width]
                else:
                    idx = torch.tensor(np.argsort(scores)[:beam_width], device=graph.device)

                array_beam_states = array_new_states[idx, :]
                best_score = float(
                    scores[idx[0]].detach() if isinstance(scores, torch.Tensor) else scores[idx[0].item()]
                )
                debug_scores[i_step] = best_score

                if verbose >= 2:
                    print(f"Step {i_step}, best score: {best_score:.2f}.")
            else:
                array_beam_states = array_new_states

            t_predict = time.time() - t_predict

            # Verbose output.
            if verbose >= 10 and (i_step - 1) % 10 == 0:
                t_full_step = time.time() - t_full_step
                print(f"Step {i_step}, beam size: {array_beam_states.shape[0]}.")

            if verbose >= 100 and (i_step - 1) % 15 == 0:
                t_full_step = time.time() - t_full_step
                print(
                    f"Time: {time.time() - t0:.1f}s, t_moves: {t_moves:.3f}s, t_hash: {t_hash:.3f}s, "
                    f"t_isin: {t_isin:.3f}s, t_unique_els: {t_unique_els:.3f}s, t_full_step: {t_full_step:.3f}s"
                )

        # Path not found.
        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")

        return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)
