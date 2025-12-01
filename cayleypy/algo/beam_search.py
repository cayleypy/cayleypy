"""Beam search algorithm for Cayley graphs."""

import time
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from .beam_search_result import BeamSearchResult
from ..bfs_result import BfsResult
from ..cayley_graph_def import AnyStateType
from ..predictor import Predictor
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


def _check_path_found(hashes, bfs_layers_hashes):
    for j, layer in enumerate(bfs_layers_hashes):
        if torch.any(isin_via_searchsorted(layer, hashes)):
            return j
    return -1


def _restore_path(
    found_layer_id: int,
    _new_hashes: torch.Tensor,
    _new_states: torch.Tensor,
    graph: "CayleyGraph",
    restore_path_hashes: list,
    bfs_layers_hashes: list,
    bfs_result_for_mitm: BfsResult,
) -> Optional[list[int]]:
    if found_layer_id == 0:
        return graph.restore_path(restore_path_hashes, graph.central_state)
    assert bfs_result_for_mitm is not None
    mask = isin_via_searchsorted(_new_hashes, bfs_layers_hashes[found_layer_id].to(graph.device))
    assert torch.any(mask), "No intersection in Meet-in-the-middle."
    middle_state = graph.decode_states(_new_states[mask.nonzero()[0].item()].reshape((1, -1)))
    path1 = graph.restore_path(restore_path_hashes, middle_state)
    path2 = graph.find_path_from(middle_state, bfs_result_for_mitm.to_device(graph.device))
    assert path2 is not None
    return path1 + path2


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
        path_device: Union[str, torch.device] = "auto",
        hashed_neigbourhood: Optional[Union[BfsResult, int]] = None,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to dest state using Beam Search algorithm.

        The following beam search modes are supported:

          * "simple" - classic beam search algorithm that finds paths from start state to central state.
            Uses meet-in-the-middle optimization if `hashed_neigbourhood` is provided.
            Supports path restoration if `return_path=True`.
          * "advanced" - enhanced beam search with non-backtracking capabilities.
            Supports configurable history depth to avoid revisiting states.
            Uses PyTorch for efficient batch processing.

        :param start_state: State from which to start search.
        :param destination_state: Target state to find. Defaults to central state for "simple" mode.
        :param beam_mode: Type of beam search (see above). Defaults to "simple".
        :param predictor: A heuristic that estimates scores for states (lower score = closer to dest).
          Defaults to Hamming distance heuristic.
        :param beam_width: Width of the beam (how many "best" states we consider at each step).
        :param max_steps: Maximum number of search steps/iterations before giving up.
        :param history_depth: For "advanced" mode, how many previous levels to remember and ban from revisiting.
        :param return_path: For "simple" mode, whether to return path (consumes much more memory if True).
        :param path_device: Device to store the path on.
        :param hashed_neigbourhood: BfsResult with pre-computed neighborhood of central state to compute for
            meet-in-the-middle modification of Beam Search. Beam search will terminate when any of states in that
            neighborhood is encountered. Defaults to None, which means no meet-in-the-middle (i.e. only search for the
            central state).
            OR
            int radius of BfsResult to be pre-computed on the go
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
                path_device=path_device,
                hashed_neigbourhood=hashed_neigbourhood,
            )
        elif beam_mode == "advanced":
            return self.search_advanced(
                start_state=start_state,
                destination_state=destination_state,
                predictor=predictor,
                beam_width=beam_width,
                max_steps=max_steps,
                return_path=return_path,
                path_device=path_device,
                hashed_neigbourhood=hashed_neigbourhood,
                history_depth=history_depth,
                verbose=verbose,
            )
        elif beam_mode == "iterated":
            return self.search_iterated(
                start_state=start_state,
                destination_state=destination_state,
                predictor=predictor,
                beam_width=beam_width,
                max_steps=max_steps,
                return_path=return_path,
                path_device=path_device,
                hashed_neigbourhood=hashed_neigbourhood,
                history_depth=history_depth,
                verbose=verbose,
            )
        else:
            raise ValueError("Unknown beam_mode:", beam_mode)

    def search_simple(
        self,
        start_state: AnyStateType,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Union[str, torch.device] = "auto",
        hashed_neigbourhood: Optional[Union[BfsResult, int]] = None,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to central state using simple Beam Search algorithm.

        :param start_state: State from which to start search.
        :param predictor: A heuristic that estimates scores for states (lower score = closer to center).
          Defaults to Hamming distance heuristic.
        :param beam_width: Width of the beam (how many "best" states we consider at each step).
        :param max_steps: Maximum number of iterations before giving up.
        :param return_path: Whether to return path (consumes much more memory if True).
        :param path_device: Device to store the path on.
        :param hashed_neigbourhood: BfsResult with pre-computed neighborhood of central state to compute for
            meet-in-the-middle modification of Beam Search. Beam search will terminate when any of states in that
            neighborhood is encountered. Defaults to None, which means no meet-in-the-middle (i.e. only search for the
            central state).
            OR
            int radius of BfsResult to be pre-computed on the go
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # Initialize predictor if not provided.
        if predictor is None:
            predictor = Predictor(graph, "hamming")

        # Use central state as a dest state.
        destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == 'auto':
            path_device = 'cpu' if return_path else graph.device

        if return_path:
            restore_path_hashes = [
                beam_hashes.to(path_device),
            ]

        # Check if start state is already the dest.
        if torch.any(beam_hashes == dest_hashes):
            return BeamSearchResult(True, 0, [], debug_scores, graph.definition)

        # Precompute meet in the middle \ destination state neighborhood hashing optimization.
        bfs_result_for_mitm: BfsResult

        hashed_neigbourhood = 0 if hashed_neigbourhood is None else hashed_neigbourhood

        if isinstance(hashed_neigbourhood, int):
            bfs_result_for_mitm = graph.bfs(
                start_states=destination_state, max_diameter=hashed_neigbourhood, return_all_hashes=True
            ).to_device(path_device)
        else:
            bfs_result_for_mitm = hashed_neigbourhood.to_device(path_device)
        assert bfs_result_for_mitm.graph == graph.definition
        bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Checks if any of `hashes` are in neighborhood of the central state.
        # Returns the number of the first layer where intersection was found, or -1 if not found.
        _new_states: torch.Tensor
        _new_hashes: torch.Tensor

        # Main beam search cycle.
        for i_step in range(1, max_steps + 1):
            # Create new states by applying all generators.
            _new_states = graph.get_neighbors(beam_states)
            # Ensure it's 2D: (n_states, state_size).
            if _new_states.dim() == 1:
                _new_states = _new_states.unsqueeze(0)
            elif _new_states.dim() > 2:
                _new_states = _new_states.flatten(end_dim=1)

            # Take only unique states.
            _new_states, _new_hashes = graph.get_unique_states(_new_states)

            # Check if dest state is found.
            bfs_layer_id = _check_path_found(_new_hashes.to(path_device), bfs_layers_hashes)
            if bfs_layer_id != -1:
                # Path found.
                path = None
                if return_path:
                    path = _restore_path(
                        bfs_layer_id,
                        _new_hashes,
                        _new_states,
                        graph,
                        restore_path_hashes,
                        bfs_layers_hashes,
                        bfs_result_for_mitm,
                    )
                return BeamSearchResult(True, i_step + bfs_layer_id, path, debug_scores, graph.definition)

            # Pick `beam_width` states with lowest scores.
            if _new_states.shape[0] > beam_width:
                scores = predictor(graph.decode_states(_new_states))
                vals, idx = torch.topk(scores, k=min(beam_width, len(scores)), largest=False, sorted=True)
                best_score = float(vals[0])

                beam_states = _new_states[idx, :]
                beam_hashes = _new_hashes[idx]

                debug_scores[i_step] = best_score

                if graph.verbose >= 2:
                    print(f"Iteration {i_step}, best score {best_score}.")
            else:
                beam_states = _new_states
                beam_hashes = _new_hashes

            if return_path:
                restore_path_hashes.append(beam_hashes.to(path_device))

        # Path not found.
        return BeamSearchResult(False, 0, None, debug_scores, graph.definition)

    def search_advanced(
        self,
        start_state: AnyStateType,
        destination_state: Optional[AnyStateType] = None,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Union[str, torch.device] = "auto",
        history_depth: int = 0,
        hashed_neigbourhood: Optional[Union[BfsResult, int]] = None,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Advanced beam search using PyTorch with non-backtracking capabilities.

        This method implements an improved beam search algorithm that supports:
        - Non-backtracking constraints (avoiding revisiting states)
        - Batch processing for efficiency
        - Configurable history depth for state banning

        :param start_state: State from which to start search.
        :param destination_state: Target state to find. Defaults to central state.
        :param predictor: Predictor object for scoring states. If None, uses Hamming distance.
        :param beam_width: Width of the beam (how many best states to consider).
        :param max_steps: Maximum number of search steps.
        :param return_path: Whether to return path (consumes much more memory if True).
        :param path_device: Device to store the path on.
        :param history_depth: How many previous levels to remember and ban from revisiting.
        :param batch_size: Batch size for model predictions. - UNUSED FOR NOW BUT WILL BE USED LATER
        :param hashed_neigbourhood: BfsResult with pre-computed neighborhood of central state to compute for
            meet-in-the-middle modification of Beam Search. Beam search will terminate when any of states in that
            neighborhood is encountered. Defaults to None, which means no meet-in-the-middle (i.e. only search for the
            central state).
            OR
            int radius of BfsResult to be pre-computed on the go
        :param verbose: Verbosity level (0=quiet, 1=basic, 10=detailed, 100=profiling).
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # Initialize predictor if not provided.
        if predictor is None:
            predictor = Predictor(graph, "hamming")

        # Use central state as dest if not specified.
        if destination_state is None:
            destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == 'auto':
            path_device = 'cpu' if return_path else graph.device

        if return_path:
            restore_path_hashes = [
                beam_hashes.to(path_device),
            ]

        # Check if start state is already the dest.
        if torch.any(beam_hashes == dest_hashes):
            return BeamSearchResult(True, 0, [], debug_scores, graph.definition)

        # Precompute meet in the middle \ destination state neighborhood hashing optimization.
        # Precompute meet in the middle \ destination state neighborhood hashing optimization.
        bfs_result_for_mitm: BfsResult

        hashed_neigbourhood = 0 if hashed_neigbourhood is None else hashed_neigbourhood

        if isinstance(hashed_neigbourhood, int):
            bfs_result_for_mitm = graph.bfs(
                start_states=destination_state, max_diameter=hashed_neigbourhood, return_all_hashes=True
            ).to_device(path_device)
        else:
            bfs_result_for_mitm = hashed_neigbourhood.to_device(path_device)
        assert bfs_result_for_mitm.graph == graph.definition
        bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Initialize hash storage for non-backtracking.
        if history_depth > 0:
            nonbacktrack_hashes = beam_hashes.expand(beam_width * graph.definition.n_generators, history_depth).clone()
            i_cyclic_index_for_hash_storage = 0

        # Checks if any of `hashes` are in neighborhood of the central state.
        # Returns the number of the first layer where intersection was found, or -1 if not found.
        _new_states: torch.Tensor
        _new_hashes: torch.Tensor

        # Main beam search cycle.
        t0 = time.time()
        for i_step in range(1, max_steps + 1):
            t_moves = t_isin = t_unique_els = 0.0
            t_full_step = time.time()

            # Create new states by applying all generators.
            t1 = time.time()
            _new_states = graph.get_neighbors(beam_states)
            # Ensure it's 2D: (n_states, state_size).
            if _new_states.dim() == 1:
                _new_states = _new_states.unsqueeze(0)
            elif _new_states.dim() > 2:
                _new_states = _new_states.flatten(end_dim=1)
            t_moves += time.time() - t1

            # Take only unique states.
            t1 = time.time()
            _new_states, _new_hashes = graph.get_unique_states(_new_states)
            t_unique_els += time.time() - t1

            # Check if dest state is found.
            bfs_layer_id = _check_path_found(_new_hashes.to(path_device), bfs_layers_hashes)
            if bfs_layer_id != -1:
                # Path found.
                path = None
                if return_path:
                    path = _restore_path(
                        bfs_layer_id,
                        _new_hashes,
                        _new_states,
                        graph,
                        restore_path_hashes,
                        bfs_layers_hashes,
                        bfs_result_for_mitm,
                    )
                return BeamSearchResult(True, i_step + bfs_layer_id, path, debug_scores, graph.definition)

            # Non-backtracking: forbid visiting states visited before.
            if history_depth > 0:

                t1 = time.time()
                mask_new = ~torch.isin(_new_hashes, nonbacktrack_hashes.view(-1), assume_unique=False)
                t_isin += time.time() - t1
                mask_new_sum = mask_new.sum().item()

                if mask_new_sum > 0:

                    # Update hash storage.
                    i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth
                    i_tmp = len(_new_hashes)
                    nonbacktrack_hashes[:i_tmp, i_cyclic_index_for_hash_storage] = _new_hashes

                    # Filter by nonbacktrack.
                    _new_states = _new_states[mask_new, :]
                    _new_hashes = _new_hashes[mask_new]

                else:
                    if verbose >= 1:
                        print(f"Cannot find new states at step {i_step}.")
                    return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

            # Estimate states and select top beam_width ones.
            t_predict = time.time()
            if _new_states.shape[0] > beam_width:
                # Score states using predictor.
                scores = predictor(graph.decode_states(_new_states))

                # Select best states.
                if isinstance(scores, torch.Tensor):
                    vals, idx = torch.topk(scores, k=min(beam_width, len(scores)), largest=False, sorted=True)
                    best_score = float(vals[0])
                else:
                    idx = torch.tensor(np.argsort(scores)[:beam_width], device=graph.device)
                    best_score = float(scores[idx[0].item()])

                beam_states = _new_states[idx, :]
                beam_hashes = _new_hashes[idx]

                debug_scores[i_step] = best_score

                if verbose >= 2:
                    print(f"Step {i_step}, best score: {best_score:.2f}.")
            else:
                beam_states = _new_states
                beam_hashes = _new_hashes

                if verbose >= 2:
                    print(f"Step {i_step}, not scored cause beam_width is big enough.")

            if return_path:
                restore_path_hashes.append(beam_hashes.to(path_device))

            t_predict = time.time() - t_predict

            # Verbose output.
            if verbose >= 10 and (i_step - 1) % 10 == 0:
                t_full_step = time.time() - t_full_step
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if verbose >= 100 and (i_step - 1) % 15 == 0:
                t_full_step = time.time() - t_full_step
                print(
                    f"Time: {time.time() - t0:.1f}s, t_moves: {t_moves:.3f}s, "  # t_hash: {t_hash:.3f}s,
                    f"t_isin: {t_isin:.3f}s, t_unique_els: {t_unique_els:.3f}s, t_full_step: {t_full_step:.3f}s"
                )

        # Path not found.
        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")

        return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)

    def search_iterated(
        self,
        start_state: AnyStateType,
        destination_state: Optional[AnyStateType] = None,
        *,
        predictor: Optional[Predictor] = None,
        beam_width: int = 1000,
        max_steps: int = 1000,
        return_path: bool = False,
        path_device: Union[str, torch.device] = "auto",
        history_depth: int = 0,
        hashed_neigbourhood: Optional[Union[BfsResult, int]] = None,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Advanced beam search using PyTorch with non-backtracking capabilities.

        This method implements an improved beam search algorithm that supports:
        - Non-backtracking constraints (avoiding revisiting states)
        - Batch processing for efficiency
        - Configurable history depth for state banning

        :param start_state: State from which to start search.
        :param destination_state: Target state to find. Defaults to central state.
        :param predictor: Predictor object for scoring states. If None, uses Hamming distance.
        :param beam_width: Width of the beam (how many best states to consider).
        :param max_steps: Maximum number of search steps.
        :param return_path: Whether to return path (consumes much more memory if True) or on which device to store it.
        :param path_device: Device to store the path on.
        :param history_depth: How many previous levels to remember and ban from revisiting.
        :param batch_size: Batch size for model predictions. - UNUSED FOR NOW BUT WILL BE USED LATER
        :param hashed_neigbourhood: BfsResult with pre-computed neighborhood of central state to compute for
            meet-in-the-middle modification of Beam Search. Beam search will terminate when any of states in that
            neighborhood is encountered. Defaults to None, which means no meet-in-the-middle (i.e. only search for the
            central state).
            OR
            int radius of BfsResult to be pre-computed on the go
        :param verbose: Verbosity level (0=quiet, 1=basic, 10=detailed, 100=profiling).
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # For now iterated beam search don't works with matrix groups.
        assert graph.definition.is_permutation_group()

        beam_width_part = beam_width // graph.definition.n_generators

        accm_states = torch.zeros((beam_width, graph.definition.state_size), dtype=graph.dtype, device=graph.device)
        accm_hashes = torch.zeros((beam_width,), dtype=torch.int64, device=graph.device)

        scores = None
        best_score = 1e6

        # Initialize predictor if not provided.
        if predictor is None:
            predictor = Predictor(graph, "hamming")

        # Use central state as dest if not specified.
        if destination_state is None:
            destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == 'auto':
            path_device = 'cpu' if return_path else graph.device

        if return_path:
            restore_path_hashes = [
                beam_hashes.to(path_device),
            ]

        # Check if start state is already the dest.
        if torch.any(beam_hashes == dest_hashes):
            return BeamSearchResult(True, 0, [], debug_scores, graph.definition)

        # Precompute meet in the middle \ destination state neighborhood hashing optimization.
        # Precompute meet in the middle \ destination state neighborhood hashing optimization.
        bfs_result_for_mitm: BfsResult

        hashed_neigbourhood = 0 if hashed_neigbourhood is None else hashed_neigbourhood

        if isinstance(hashed_neigbourhood, int):
            bfs_result_for_mitm = graph.bfs(
                start_states=destination_state, max_diameter=hashed_neigbourhood, return_all_hashes=True
            ).to_device(path_device)
        else:
            bfs_result_for_mitm = hashed_neigbourhood.to_device(path_device)
        assert bfs_result_for_mitm.graph == graph.definition
        bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Initialize hash storage for non-backtracking.
        if history_depth > 0:
            nonbacktrack_hashes = beam_hashes.expand(beam_width * graph.definition.n_generators, history_depth).clone()
            i_cyclic_index_for_hash_storage = 0

        # Checks if any of `hashes` are in neighborhood of the central state.
        # Returns the number of the first layer where intersection was found, or -1 if not found.
        _new_states_chunk: torch.Tensor
        _new_hashes_chunk: torch.Tensor

        # Main beam search cycle.
        t0 = time.time()
        for i_step in range(1, max_steps + 1):
            t_moves = t_isin = t_unique_els = t_predict = 0.0
            t_full_step = time.time()

            _chunk_idx = 0
            i_tmp = 0

            if history_depth > 0:
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth

            accm_states.fill_(0)
            accm_hashes.fill_(0)

            # Create new states by applying all generators one by one.
            for _new_states_chunk in graph.get_neighbors_generator(beam_states):
                # Ensure it's 2D: (n_states, state_size).
                t1 = time.time()
                if _new_states_chunk.dim() == 1:
                    _new_states_chunk = _new_states_chunk.unsqueeze(0)
                elif _new_states_chunk.dim() > 2:
                    _new_states_chunk = _new_states_chunk.flatten(end_dim=1)
                t_moves += time.time() - t1

                _new_hashes_chunk = graph.hasher.make_hashes(_new_states_chunk)
                _new_hashes_chunk, idx = torch.sort(_new_hashes_chunk, stable=True)
                _new_states_chunk = _new_states_chunk[idx, :]

                # Skip already generated states instead of deduplication at the end.
                if _chunk_idx > 0:
                    mask_new = ~torch.isin(_new_hashes_chunk, accm_hashes.view(-1), assume_unique=False)
                    mask_new_sum = mask_new.sum().item()

                    if mask_new_sum > 0:
                        _new_states_chunk = _new_states_chunk[mask_new, :]
                        _new_hashes_chunk = _new_hashes_chunk[mask_new]
                    else:
                        continue

                # Check if dest state is found.
                bfs_layer_id = _check_path_found(_new_hashes_chunk.to(path_device), bfs_layers_hashes)
                if bfs_layer_id != -1:
                    # Path found.
                    path = None
                    if return_path:
                        path = _restore_path(
                            bfs_layer_id,
                            _new_hashes_chunk,
                            _new_states_chunk,
                            graph,
                            restore_path_hashes,
                            bfs_layers_hashes,
                            bfs_result_for_mitm,
                        )
                    return BeamSearchResult(True, i_step + bfs_layer_id, path, debug_scores, graph.definition)

                # Non-backtracking: forbid visiting states visited before.
                if history_depth > 0:

                    t1 = time.time()
                    mask_new = ~torch.isin(_new_hashes_chunk, nonbacktrack_hashes.view(-1), assume_unique=False)
                    t_isin += time.time() - t1
                    mask_new_sum = mask_new.sum().item()

                    if mask_new_sum > 0:

                        # Update hash storage.
                        nonbacktrack_hashes[i_tmp : i_tmp + len(_new_hashes_chunk), i_cyclic_index_for_hash_storage] = (
                            _new_hashes_chunk
                        )
                        i_tmp += len(_new_hashes_chunk)

                        _new_states_chunk = _new_states_chunk[mask_new, :]
                        _new_hashes_chunk = _new_hashes_chunk[mask_new]
                    else:
                        continue

                # Estimate states and select top beam_width ones.
                t1 = time.time()
                if _new_states_chunk.shape[0] > beam_width_part:
                    # Score states using predictor.
                    scores = predictor(graph.decode_states(_new_states_chunk))

                    # Select best states.
                    if isinstance(scores, torch.Tensor):
                        vals, idx = torch.topk(scores, k=min(beam_width_part, len(scores)), largest=False, sorted=True)
                        best_score = float(vals[0])
                    else:
                        idx = torch.tensor(np.argsort(scores)[:beam_width_part], device=graph.device)
                        best_score = float(scores[idx[0].item()])

                    _new_states_chunk = _new_states_chunk[idx, :]
                    _new_hashes_chunk = _new_hashes_chunk[idx]

                    if (i_step not in debug_scores) or (best_score < debug_scores[i_step]):
                        debug_scores[i_step] = best_score

                if _new_states_chunk.shape[0] > 0:
                    accm_states[_chunk_idx : _chunk_idx + _new_states_chunk.shape[0], :] = _new_states_chunk
                    accm_hashes[_chunk_idx : _chunk_idx + _new_states_chunk.shape[0]] = _new_hashes_chunk

                    _chunk_idx += _new_states_chunk.shape[0]

                t_predict += time.time() - t1

            if _chunk_idx == 0:
                if verbose >= 1:
                    print(f"Cannot find new states at step {i_step}.")
                return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

            beam_states = accm_states[:_chunk_idx, :].clone()
            beam_hashes = accm_hashes[:_chunk_idx].clone()

            _chunk_idx = 0

            if verbose >= 2:
                if i_step in debug_scores:
                    print(f"Step {i_step}, best score: {debug_scores[i_step]:.2f}.")
                else:
                    print(f"Step {i_step}, not scored cause beam_width is big enough.")

            if return_path:
                restore_path_hashes.append(beam_hashes.to(path_device))

            # Verbose output.
            if verbose >= 10 and (i_step - 1) % 10 == 0:
                t_full_step = time.time() - t_full_step
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if verbose >= 100 and (i_step - 1) % 15 == 0:
                t_full_step = time.time() - t_full_step
                print(
                    f"Time: {time.time() - t0:.1f}s, t_moves: {t_moves:.3f}s, "  # t_hash: {t_hash:.3f}s,
                    f"t_isin: {t_isin:.3f}s, t_unique_els: {t_unique_els:.3f}s, t_full_step: {t_full_step:.3f}s"
                )

        # Path not found.
        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")

        return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)
