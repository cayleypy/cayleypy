"""Beam search algorithm for Cayley graphs."""

import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from .beam_search_result import BeamSearchResult
from ..bfs_result import BfsResult
from ..cayley_graph_def import AnyStateType
from ..predictor import Predictor
from ..torch_utils import isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


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
            )
        elif beam_mode == "advanced":
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
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        graph = self.graph
        if predictor is None:
            predictor = Predictor(graph, "hamming")

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

        for i in range(max_steps):
            # Create states on the next layer.
            layer2, layer2_hashes = graph.get_unique_states(graph.get_neighbors(layer1))

            bfs_layer_id = _check_path_found(layer2_hashes)
            if bfs_layer_id != -1:
                # Path found.
                path = _restore_path(bfs_layer_id)
                return BeamSearchResult(True, i + bfs_layer_id + 1, path, debug_scores, graph.definition)

            # Pick `beam_width` states with lowest scores.
            if len(layer2) >= beam_width:
                scores = predictor(graph.decode_states(layer2))
                idx = torch.argsort(scores)[:beam_width]
                layer2 = layer2[idx, :]
                layer2_hashes = layer2_hashes[idx]
                best_score = float(scores[idx[0]].detach())
                debug_scores[i] = best_score
                if graph.verbose >= 2:
                    print(f"Iteration {i}, best score {best_score}.")

            layer1 = layer2
            layer1_hashes = layer2_hashes
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
