"""Beam search algorithm for Cayley graphs."""

import time
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from .beam_search_result import BeamSearchResult
from ..bfs_result import BfsResult
from ..cayley_graph_def import AnyStateType
from ..predictor import Predictor
from ..torch_utils import TorchHashSet, isin_via_searchsorted

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


def _cuda_sync() -> None:
    """Synchronize the GPU stream so wall-clock timers reflect actual execution.

    Without this, `time.time()` around asynchronous GPU ops measures kernel
    *launch* time, not execution time (the exact bug AGENTS.md §10 warns about
    for the Kaggle benchmark script). Only called when `verbose >= 100`, so the
    sync cost never pollutes benchmark timings (benchmarks run at verbose=0).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _BeamSearchProfile:
    """Accumulates per-region GPU-synced timings for `verbose >= 100` profiling.

    Each region is timed with `time.time()` bracketed by `_cuda_sync()` so the
    measurement reflects actual GPU execution (not kernel launch time). The
    accumulator tracks per-region totals across all steps and chunks; the
    `verbose >= 100` print line reports both the running total and the
    per-step delta so dominant regions are visible at a glance.

    Regions tracked (iterated mode; advanced mode tracks the subset that applies):
      t_moves   — dim-check / unsqueeze (trivial; the actual neighbor application
                  is inside `get_neighbors_generator` and not separately timed
                  here — use the torch.profiler trace for that level of detail).
      t_hash    — `hasher.make_hashes` (was UNTIMED in the original code).
      t_sort    — `torch.sort` of chunk hashes (was UNTIMED).
      t_dedup   — accumulator `torch.isin` + mask application (was UNTIMED).
      t_check   — `_check_path_found` (was UNTIMED).
      t_isin    — non-backtracking isin loop + mask application.
      t_predict — predictor call + topk + accumulator write.
    """

    __slots__ = ("moves", "hash", "sort", "dedup", "check", "isin", "predict", "step_start")

    def __init__(self) -> None:
        self.moves: float = 0.0
        self.hash: float = 0.0
        self.sort: float = 0.0
        self.dedup: float = 0.0
        self.check: float = 0.0
        self.isin: float = 0.0
        self.predict: float = 0.0
        self.step_start: float = 0.0

    def reset_step(self) -> None:
        """Start a new step: snapshot the per-step accumulator baseline."""
        self.step_start = time.time()
        self.moves = self.hash = self.sort = self.dedup = 0.0
        self.check = self.isin = self.predict = 0.0

    def format_line(self, i_step: int, t0: float) -> str:
        """Format the verbose=100 profiling line for the current step."""
        total = time.time() - self.step_start
        return (
            f"  step {i_step}: total={total:.3f}s "
            f"moves={self.moves:.3f} hash={self.hash:.3f} sort={self.sort:.3f} "
            f"dedup={self.dedup:.3f} check={self.check:.3f} "
            f"isin={self.isin:.3f} predict={self.predict:.3f} "
            f"(elapsed={time.time() - t0:.1f}s)"
        )


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
    if not torch.any(mask):
        raise ValueError("No intersection in Meet-in-the-middle.")
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
        memory_cleanup: bool = False,
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
          * "iterated" - iterated beam search (per-generator chunking). Best path-finding
            quality at large beams (preserves per-generator fairness via per-chunk topk).
          * "iterated_batched" - batched variant of iterated (Phase 4). Materializes all
            neighbors at once (ONE hash+sort+dedup+predictor), then per-generator topk
            with origin tracking. Faster than "iterated" at medium beams (regime B,
            <= ~2^21 for cube555) by eliminating per-chunk Python/launch overhead.
            Falls back to "iterated" if the memory gate trips (n_gens x bw x state_size
            > 0.6 x device memory). Opt-in; fairness measured per group.

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
          At level 100, each step prints a GPU-synced per-region timing breakdown
          (moves/hash/sort/dedup/check/isin/predict). The sync brackets add
          overhead — NEVER use verbose>=100 in benchmarks; use it only for
          one-shot profiling (see kaggle_benchmarks/perf/run.py `run_profiling`).
          For a Chrome-trace flame graph (lower friction, no per-step syncs),
          wrap your `graph.beam_search(...)` call in `torch.profiler.profile`
          externally and export with `prof.export_chrome_trace(path)`.
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
                memory_cleanup=memory_cleanup,
                verbose=verbose,
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
                memory_cleanup=memory_cleanup,
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
                memory_cleanup=memory_cleanup,
                verbose=verbose,
            )
        elif beam_mode == "iterated_batched":
            return self.search_iterated_batched(
                start_state=start_state,
                destination_state=destination_state,
                predictor=predictor,
                beam_width=beam_width,
                max_steps=max_steps,
                return_path=return_path,
                path_device=path_device,
                hashed_neigbourhood=hashed_neigbourhood,
                history_depth=history_depth,
                memory_cleanup=memory_cleanup,
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
        memory_cleanup: bool = False,
        verbose: int = 0,
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
        :param verbose: Verbosity level (0=quiet, 1=basic, 10=detailed, 100=profiling).
          At level 100, each step prints a GPU-synced per-region timing breakdown
          (moves/hash/check/predict). The sync brackets add overhead — NEVER use
          verbose>=100 in benchmarks; use it only for one-shot profiling.
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # Initialize predictor if not provided.
        if predictor is None:
            _predictor = Predictor(graph, "hamming")
        elif isinstance(predictor, Predictor):
            _predictor = predictor
        else:
            _predictor = Predictor(graph, predictor)

        # Use central state as a dest state.
        destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == "auto":
            path_device = "cpu" if return_path else graph.device

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
        if bfs_result_for_mitm.graph != graph.definition:
            raise ValueError("Graph from bfs_result_for_mitm must be the same.")
        bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        _new_states: torch.Tensor
        _new_hashes: torch.Tensor

        # Main beam search cycle.
        t0 = time.time()
        profile = _BeamSearchProfile() if verbose >= 100 else None
        for i_step in range(1, max_steps + 1):
            if profile is not None:
                _cuda_sync()
                profile.reset_step()

            # Create new states by applying all generators.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _new_states = graph.get_neighbors(beam_states)
            # Ensure it's 2D: (n_states, state_size).
            if _new_states.dim() == 1:
                _new_states = _new_states.unsqueeze(0)
            elif _new_states.dim() > 2:
                _new_states = _new_states.flatten(end_dim=1)
            if profile is not None:
                _cuda_sync()
                profile.moves += time.time() - t1

            # Take only unique states (bundles make_hashes + sort + dedup inside get_unique_states).
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _new_states, _new_hashes = graph.get_unique_states(_new_states)
            if profile is not None:
                _cuda_sync()
                profile.hash += time.time() - t1

            # Check if dest state is found.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            bfs_layer_id = _check_path_found(_new_hashes.to(path_device), bfs_layers_hashes)
            if profile is not None:
                _cuda_sync()
                profile.check += time.time() - t1
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
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            if _new_states.shape[0] > beam_width:
                scores = _predictor(graph.decode_states(_new_states))
                vals, idx = torch.topk(scores, k=min(beam_width, len(scores)), largest=False, sorted=True)
                best_score = float(vals[0])

                beam_states = _new_states[idx, :]
                beam_hashes = _new_hashes[idx]

                debug_scores[i_step] = best_score

                if verbose >= 2:
                    print(f"Iteration {i_step}, best score {best_score}.")
            else:
                beam_states = _new_states
                beam_hashes = _new_hashes

                if verbose >= 2:
                    print(f"Iteration {i_step}, not scored cause beam_width is big enough.")
            if profile is not None:
                _cuda_sync()
                profile.predict += time.time() - t1

            if return_path:
                restore_path_hashes.append(beam_hashes.to(path_device))

            if memory_cleanup:
                graph.free_memory()

            if verbose >= 10 and (i_step - 1) % 10 == 0:
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if profile is not None:
                _cuda_sync()
                print(profile.format_line(i_step, t0))

        # Path not found.
        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")

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
        memory_cleanup: bool = False,
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
          At level 100, each step prints a GPU-synced per-region timing breakdown
          (moves/hash/sort/dedup/check/isin/predict). The sync brackets add
          overhead — NEVER use verbose>=100 in benchmarks; use it only for
          one-shot profiling.
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # Initialize predictor if not provided.
        if predictor is None:
            _predictor = Predictor(graph, "hamming")
        elif isinstance(predictor, Predictor):
            _predictor = predictor
        else:
            _predictor = Predictor(graph, predictor)

        # Use central state as dest if not specified.
        if destination_state is None:
            destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == "auto":
            path_device = "cpu" if return_path else graph.device

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
        if bfs_result_for_mitm.graph != graph.definition:
            raise ValueError("Graph from bfs_result_for_mitm must be the same.")
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
        profile = _BeamSearchProfile() if verbose >= 100 else None
        for i_step in range(1, max_steps + 1):
            if profile is not None:
                _cuda_sync()
                profile.reset_step()

            # Create new states by applying all generators.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _new_states = graph.get_neighbors(beam_states)
            # Ensure it's 2D: (n_states, state_size).
            if _new_states.dim() == 1:
                _new_states = _new_states.unsqueeze(0)
            elif _new_states.dim() > 2:
                _new_states = _new_states.flatten(end_dim=1)
            if profile is not None:
                _cuda_sync()
                profile.moves += time.time() - t1

            # Take only unique states (internally: make_hashes + sort + dedup).
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _new_states, _new_hashes = graph.get_unique_states(_new_states)
            if profile is not None:
                _cuda_sync()
                # In advanced mode, hash+sort+dedup are inside get_unique_states;
                # the iterated mode splits them (see search_iterated). Time the
                # whole call as t_hash here — use the profiler trace for finer detail.
                profile.hash += time.time() - t1

            # Check if dest state is found.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            bfs_layer_id = _check_path_found(_new_hashes.to(path_device), bfs_layers_hashes)
            if profile is not None:
                _cuda_sync()
                profile.check += time.time() - t1
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
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()

                mask_new = torch.ones_like(_new_hashes, dtype=torch.bool)
                for j in range(nonbacktrack_hashes.shape[1]):
                    mask_new *= ~torch.isin(_new_hashes, nonbacktrack_hashes[:, j], assume_unique=False)

                # Update hash storage.
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth
                i_tmp = len(_new_hashes)
                nonbacktrack_hashes[:i_tmp, i_cyclic_index_for_hash_storage] = _new_hashes

                # Apply mask unconditionally (Task 1.3): drops the .item()
                # GPU→CPU sync guard. Empty result is handled by the shape[0]==0
                # early-exit below.
                _new_states = _new_states[mask_new, :]
                _new_hashes = _new_hashes[mask_new]

                if profile is not None:
                    _cuda_sync()
                    profile.isin += time.time() - t1

            if _new_hashes.shape[0] == 0:
                if verbose >= 1:
                    print(f"Cannot find new states at step {i_step}.")
                return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

            # Estimate states and select top beam_width ones.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            if _new_states.shape[0] > beam_width:
                # Score states using predictor.
                scores = _predictor(graph.decode_states(_new_states))

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

            if memory_cleanup:
                graph.free_memory()

            if profile is not None:
                _cuda_sync()
                profile.predict += time.time() - t1

            # Verbose output.
            if verbose >= 10 and (i_step - 1) % 10 == 0:
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if profile is not None:
                _cuda_sync()
                print(profile.format_line(i_step, t0))

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
        memory_cleanup: bool = False,
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
          At level 100, each step prints a GPU-synced per-region timing breakdown
          (moves/hash/sort/dedup/check/isin/predict). The sync brackets add
          overhead — NEVER use verbose>=100 in benchmarks; use it only for
          one-shot profiling.
        :return: BeamSearchResult containing found path length and (optionally) the path itself.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # For now iterated beam search don't works with matrix groups.
        if not graph.definition.is_permutation_group():
            raise ValueError("Iterated beam search actually realized only for Permutation Groups.")

        beam_width_part = beam_width // graph.definition.n_generators

        accm_states = torch.zeros((beam_width, graph.definition.state_size), dtype=graph.dtype, device=graph.device)
        # Compact dedup set (dedup-unification): replaces the zero-padded accm_hashes
        # buffer + torch.isin. Fixes the latent zero-padding bug (hash==0 states
        # falsely deduped) and is faster (searchsorted vs torch.isin re-sort).
        # Queries via get_mask_to_remove_seen_hashes (isin_via_searchsorted on sorted
        # shards); adds via add_sorted_hashes (requires sorted input — guaranteed by
        # the sort step, with a re-sort after topk to restore hash order).
        accm_hashset = TorchHashSet()

        scores = None
        best_score = 1e6

        # Initialize predictor if not provided.
        if predictor is None:
            _predictor = Predictor(graph, "hamming")
        elif isinstance(predictor, Predictor):
            _predictor = predictor
        else:
            _predictor = Predictor(graph, predictor)

        # Use central state as dest if not specified.
        if destination_state is None:
            destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == "auto":
            path_device = "cpu" if return_path else graph.device

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
        if bfs_result_for_mitm.graph != graph.definition:
            raise ValueError("Graph from bfs_result_for_mitm must be the same.")
        bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Initialize hash storage for non-backtracking.
        # Compact representation (Task 1.6): one TorchHashSet per history-depth slot,
        # holding only the actual hashes generated each step. Replaces the old dense
        # (beam_width × n_generators, history_depth) int64 matrix which:
        #   - was preallocated to the maximum beam size (bw*ng) so any layer would
        #     fit — the start hash filling it was an artifact of `expand` from a
        #     single-element tensor, not an intentional invariant;
        #   - consumed 6.00 GB at bw=2^24, cube555, hd=2 (vs ~0.25 GB here), AND
        #   - was queried by a Python `for j in range(hd)` isin loop launching `hd`
        #     kernels per chunk (the dominant t_isin cost, 57-70% per profiling).
        # Invariants preserved:
        #   (a) `add_sorted_hashes` requires sorted input — chunk hashes are sorted
        #       at the sort step above;
        #   (b) `get_merged_sorted` collapses shards before querying (no Python loop).
        # The start hash is NOT re-added on reset: it was an artifact of the old
        # preallocation, not a correctness requirement. The seen set holds only the
        # actual hashes from the previous `history_depth` steps.
        if history_depth > 0:
            nonbacktrack_hashes: list[TorchHashSet] = [TorchHashSet() for _ in range(history_depth)]
            i_cyclic_index_for_hash_storage = 0

        # Checks if any of `hashes` are in neighborhood of the central state.
        # Returns the number of the first layer where intersection was found, or -1 if not found.
        _new_states_chunk: torch.Tensor
        _new_hashes_chunk: torch.Tensor

        # Main beam search cycle.
        t0 = time.time()
        profile = _BeamSearchProfile() if verbose >= 100 else None
        for i_step in range(1, max_steps + 1):
            if profile is not None:
                _cuda_sync()
                profile.reset_step()

            _chunk_idx = 0

            if history_depth > 0:
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth
                # Reset the current slot (Task 1.6): the old dense matrix did this
                # implicitly by overwriting rows in-place; the compact hash-set must
                # be cleared explicitly before adding this step's hashes.
                nonbacktrack_hashes[i_cyclic_index_for_hash_storage].data = []

            accm_states.fill_(0)
            accm_hashset.data = []

            # Create new states by applying all generators one by one.
            # clone=False: the yielded buffer is reused across generators. Safe because
            # the consumer reassigns `_new_states_chunk` via fancy-indexing at the sort
            # step below (idx = torch.sort(...); _new_states_chunk = _new_states_chunk[idx, :])
            # BEFORE the next yield — so no alias survives past the next generator's
            # write. See Task 1.5 (no-alias invariant documented in plan).
            for _new_states_chunk in graph.get_neighbors_generator(beam_states, clone=False):
                # Ensure it's 2D: (n_states, state_size).
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()
                if _new_states_chunk.dim() == 1:
                    _new_states_chunk = _new_states_chunk.unsqueeze(0)
                elif _new_states_chunk.dim() > 2:
                    _new_states_chunk = _new_states_chunk.flatten(end_dim=1)
                if profile is not None:
                    _cuda_sync()
                    profile.moves += time.time() - t1

                # Hash the chunk (was UNTIMED in the original code).
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()
                _new_hashes_chunk = graph.hasher.make_hashes(_new_states_chunk)
                if profile is not None:
                    _cuda_sync()
                    profile.hash += time.time() - t1

                # Sort by hash (was UNTIMED; preserves the sorted invariant, AGENTS.md §6).
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()
                _new_hashes_chunk, idx = torch.sort(_new_hashes_chunk, stable=True)
                _new_states_chunk = _new_states_chunk[idx, :]
                if profile is not None:
                    _cuda_sync()
                    profile.sort += time.time() - t1

                # Skip already generated states (dedup-unification): replaces
                # torch.isin against the zero-padded accm_hashes buffer with a
                # TorchHashSet query (isin_via_searchsorted on sorted shards).
                # Fixes the latent zero-padding bug (hash==0 states falsely deduped)
                # and is faster (searchsorted vs torch.isin re-sorting the full buffer).
                if _chunk_idx > 0:
                    if profile is not None:
                        _cuda_sync()
                        t1 = time.time()
                    mask_new = accm_hashset.get_mask_to_remove_seen_hashes(_new_hashes_chunk)

                    # Apply mask unconditionally (Task 1.3): the old
                    # `mask_new.sum().item() > 0` guard forced a GPU→CPU sync per
                    # chunk. Boolean indexing with all-False produces an empty
                    # tensor (shape [0, ...]), handled by downstream guards
                    # (shape[0] > beam_width_part skips topk; shape[0] > 0 skips
                    # accumulator write).
                    _new_states_chunk = _new_states_chunk[mask_new, :]
                    _new_hashes_chunk = _new_hashes_chunk[mask_new]
                    if profile is not None:
                        _cuda_sync()
                        profile.dedup += time.time() - t1

                # Check if dest state is found. (was UNTIMED; the .to() is a no-op when
                # return_path=False since path_device==graph.device — see Task 1.1 caveat.)
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()
                bfs_layer_id = _check_path_found(_new_hashes_chunk.to(path_device), bfs_layers_hashes)
                if profile is not None:
                    _cuda_sync()
                    profile.check += time.time() - t1
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
                # Compact hash-set query (Task 1.6): one `get_merged_sorted` +
                # one `isin_via_searchsorted` replaces the old `for j in range(hd)`
                # loop that launched `hd` separate `torch.isin` kernels per chunk.
                # The merged tensor is cached for the step (all chunks share it).
                if history_depth > 0:
                    if profile is not None:
                        _cuda_sync()
                        t1 = time.time()

                    # Query all history slots: union of "seen in any prior slot".
                    # Each slot is queried via its own merged sorted tensor; the
                    # number of slots == history_depth (small, e.g. 2), so this
                    # is a bounded loop (vs the old per-depth isin loop which was
                    # also hd-iterations but each did a full torch.isin).
                    mask_new = torch.ones_like(_new_hashes_chunk, dtype=torch.bool)
                    for slot in nonbacktrack_hashes:
                        mask_new &= slot.get_mask_to_remove_seen_hashes(_new_hashes_chunk)

                    # Update the current step's slot with this chunk's hashes.
                    # Precondition: `_new_hashes_chunk` is sorted (sort step above)
                    # and deduplicated against the accumulator (dedup step above),
                    # so `add_sorted_hashes` is safe.
                    nonbacktrack_hashes[i_cyclic_index_for_hash_storage].add_sorted_hashes(_new_hashes_chunk)

                    # Apply mask unconditionally (Task 1.3): drops the .item()
                    # GPU→CPU sync guard. Empty result (all-False mask) is handled
                    # by downstream shape[0] guards.
                    _new_states_chunk = _new_states_chunk[mask_new, :]
                    _new_hashes_chunk = _new_hashes_chunk[mask_new]

                    if profile is not None:
                        _cuda_sync()
                        profile.isin += time.time() - t1

                # Estimate states and select top beam_width ones.
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()
                _topk_applied = False
                if _new_states_chunk.shape[0] > beam_width_part:
                    # Score states using predictor.
                    scores = _predictor(graph.decode_states(_new_states_chunk))

                    # Select best states.
                    if isinstance(scores, torch.Tensor):
                        vals, idx = torch.topk(scores, k=min(beam_width_part, len(scores)), largest=False, sorted=True)
                        best_score = float(vals[0])
                    else:
                        idx = torch.tensor(np.argsort(scores)[:beam_width_part], device=graph.device)
                        best_score = float(scores[idx[0].item()])

                    _new_states_chunk = _new_states_chunk[idx, :]
                    _new_hashes_chunk = _new_hashes_chunk[idx]
                    _topk_applied = True

                    if (i_step not in debug_scores) or (best_score < debug_scores[i_step]):
                        debug_scores[i_step] = best_score

                if _new_states_chunk.shape[0] > 0:
                    accm_states[_chunk_idx : _chunk_idx + _new_states_chunk.shape[0], :] = _new_states_chunk

                    # Add chunk hashes to the dedup hashset (dedup-unification).
                    # Precondition: add_sorted_hashes requires sorted-by-hash input.
                    # The sort step (:789) guarantees this, BUT topk (above) reorders
                    # by score via idx, breaking the hash order. Re-sort only when
                    # topk was applied. When topk is skipped, _new_hashes_chunk is
                    # still hash-sorted (preserved through boolean-mask dedup/nonbacktrack).
                    if _topk_applied:
                        _hash_sort_idx = torch.argsort(_new_hashes_chunk, stable=True)
                        accm_hashset.add_sorted_hashes(_new_hashes_chunk[_hash_sort_idx])
                    else:
                        accm_hashset.add_sorted_hashes(_new_hashes_chunk)

                    _chunk_idx += _new_states_chunk.shape[0]

                if profile is not None:
                    _cuda_sync()
                    profile.predict += time.time() - t1

            if _chunk_idx == 0:
                if verbose >= 1:
                    print(f"Cannot find new states at step {i_step}.")
                return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

            beam_states = accm_states[:_chunk_idx, :].clone()
            # Get merged sorted hashes from the hashset (dedup-unification): replaces
            # accm_hashes[:k].clone(). The hashset owns the tensor; on next step's
            # reset (data = []), beam_hashes still holds a reference → tensor kept alive.
            beam_hashes = accm_hashset.get_merged_sorted()

            _chunk_idx = 0

            if verbose >= 2:
                if i_step in debug_scores:
                    print(f"Step {i_step}, best score: {debug_scores[i_step]:.2f}.")
                else:
                    print(f"Step {i_step}, not scored cause beam_width is big enough.")

            # beam_hashes is NOT sorted after per-generator topk (generators are
            # concatenated in gen order, each sub-block sorted by score). For path
            # restoration this is fine (uses isin membership, not order). For the
            # next step's nonbacktrack add_sorted_hashes, we need sorted — but that
            # add happens in step 5 which re-sorts via the dedup inline. So no
            # explicit re-sort needed here.
            if return_path:
                restore_path_hashes.append(beam_hashes.to(path_device))

            if memory_cleanup:
                graph.free_memory()

            # Verbose output.
            if verbose >= 10 and (i_step - 1) % 10 == 0:
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if profile is not None:
                _cuda_sync()
                print(profile.format_line(i_step, t0))

        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")

        return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)

    def search_iterated_batched(
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
        memory_cleanup: bool = False,
        verbose: int = 0,
    ) -> BeamSearchResult:
        """Batched variant of iterated beam search (Phase 4, regime B speed).

        Materializes all `n_gens x beam_width` neighbors at once (ONE hash+sort+dedup
        + ONE _check_path_found + ONE nonbacktrack check + ONE predictor call), then
        per-generator topk with origin tracking. Eliminates the per-chunk Python loop
        and per-chunk kernel launches of `search_iterated`.

        **Memory gate:** if `n_gens x beam_width x state_size > 0.6 x device_memory`,
        falls back to `search_iterated` (chunked, Phase 1 path). Regime B only
        (<= ~2^21 cube555, <= ~2^22 cube333 on a 16 GB GPU); regime A (2^24) uses
        chunked+compaction.

        **Fairness:** per-generator topk preserves per-generator slot allocation
        (`beam_width // n_gens` each), unlike advanced mode's global topk which can
        starve generators. Dedup may shift the survivor distribution, so equivalence to
        `search_iterated` is measured per group (see validation tests).

        Opt-in via `beam_mode="iterated_batched"`. All params match `search_iterated`.
        """
        debug_scores: dict[int, float] = {}

        graph = self.graph

        # Iterated modes require permutation groups (get_neighbors concatenates
        # n_gens chunks of beam_width each; the origin-tracking math relies on this).
        if not graph.definition.is_permutation_group():
            raise ValueError("Iterated batched beam search actually realized only for Permutation Groups.")

        n_generators = graph.definition.n_generators
        state_size = graph.definition.state_size
        beam_width_part = beam_width // n_generators

        # Memory gate: if materializing all n_gens x bw neighbors exceeds 0.6x device
        # memory, fall back to chunked search_iterated (Phase 1 path).
        if graph.device.type == "cuda":
            device_memory = torch.cuda.get_device_properties(graph.device).total_memory
            batched_states_bytes = n_generators * beam_width * state_size * graph.dtype.itemsize
            # Conservative: neighbors buffer (1x) + hashes (8B/elem) + dedup buffer (1x) +
            # predictor intermediate (~1x). Use 3x the states buffer as the budget estimate.
            if batched_states_bytes * 3 > 0.6 * device_memory:
                if verbose >= 1:
                    print(
                        f"Memory gate tripped: batched would need ~{batched_states_bytes * 3 / 2**30:.1f} GB "
                        f"(> 60% of {device_memory / 2**30:.1f} GB device). Falling back to chunked iterated."
                    )
                return self.search_iterated(
                    start_state=start_state,
                    destination_state=destination_state,
                    predictor=predictor,
                    beam_width=beam_width,
                    max_steps=max_steps,
                    return_path=return_path,
                    path_device=path_device,
                    history_depth=history_depth,
                    hashed_neigbourhood=hashed_neigbourhood,
                    memory_cleanup=memory_cleanup,
                    verbose=verbose,
                )

        # Initialize predictor if not provided.
        if predictor is None:
            _predictor = Predictor(graph, "hamming")
        elif isinstance(predictor, Predictor):
            _predictor = predictor
        else:
            _predictor = Predictor(graph, predictor)

        # Use central state as dest if not specified.
        if destination_state is None:
            destination_state = graph.central_state

        # Encode states.
        beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))
        _, dest_hashes = graph.get_unique_states(graph.encode_states(destination_state))

        if path_device == "auto":
            path_device = "cpu" if return_path else graph.device

        if return_path:
            restore_path_hashes = [
                beam_hashes.to(path_device),
            ]

        # Check if start state is already the dest.
        if torch.any(beam_hashes == dest_hashes):
            return BeamSearchResult(True, 0, [], debug_scores, graph.definition)

        # Precompute meet-in-the-middle neighborhood.
        bfs_result_for_mitm: BfsResult
        hashed_neigbourhood = 0 if hashed_neigbourhood is None else hashed_neigbourhood
        if isinstance(hashed_neigbourhood, int):
            bfs_result_for_mitm = graph.bfs(
                start_states=destination_state, max_diameter=hashed_neigbourhood, return_all_hashes=True
            ).to_device(path_device)
        else:
            bfs_result_for_mitm = hashed_neigbourhood.to_device(path_device)
        if bfs_result_for_mitm.graph != graph.definition:
            raise ValueError("Graph from bfs_result_for_mitm must be the same.")
        bfs_layers_hashes = bfs_result_for_mitm.layers_hashes

        # Initialize compact non-backtracking hash storage (same as search_iterated Task 1.6).
        if history_depth > 0:
            nonbacktrack_hashes: list[TorchHashSet] = [TorchHashSet() for _ in range(history_depth)]
            i_cyclic_index_for_hash_storage = 0

        # Main beam search cycle.
        t0 = time.time()
        profile = _BeamSearchProfile() if verbose >= 100 else None
        for i_step in range(1, max_steps + 1):
            if profile is not None:
                _cuda_sync()
                profile.reset_step()

            if history_depth > 0:
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth
                nonbacktrack_hashes[i_cyclic_index_for_hash_storage].data = []

            # 1. Materialize ALL neighbors at once: shape (n_gens * bw, state_size).
            # get_neighbors concatenates n_gens chunks of bw each, so origin of index i
            # is i // beam_states.shape[0].
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _new_states = graph.get_neighbors(beam_states)
            if _new_states.dim() == 1:
                _new_states = _new_states.unsqueeze(0)
            elif _new_states.dim() > 2:
                _new_states = _new_states.flatten(end_dim=1)
            _beam_size = beam_states.shape[0]
            if profile is not None:
                _cuda_sync()
                profile.moves += time.time() - t1

            # 2. Hash all neighbors at once (ONE make_hashes call vs n_gens per-chunk).
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _new_hashes = graph.hasher.make_hashes(_new_states)
            if profile is not None:
                _cuda_sync()
                profile.hash += time.time() - t1

            # 3. Inlined dedup: sort by hash, keep first occurrence of each hash.
            # Also returns unique_idx (origin tracking) — get_unique_states computes
            # this internally but does not return it, so we inline (~8 lines).
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            _hashes_sorted, _sort_idx = torch.sort(_new_hashes, stable=True)
            _dedup_mask = torch.ones(_hashes_sorted.size(0), dtype=torch.bool, device=graph.device)
            if _hashes_sorted.size(0) > 1:
                _dedup_mask[1:] = _hashes_sorted[1:] != _hashes_sorted[:-1]
            _unique_idx = _sort_idx[_dedup_mask]  # indices into the ORIGINAL n_gens*bw array
            _new_states = _new_states[_unique_idx]
            _new_hashes = _hashes_sorted[_dedup_mask]  # already sorted (from the sort above)
            # Origin: which generator produced each surviving state.
            _gen_origin = _unique_idx // _beam_size
            if profile is not None:
                _cuda_sync()
                profile.sort += time.time() - t1
                profile.dedup += time.time() - t1  # dedup is fused with sort here

            # 4. Check if dest state is found.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            bfs_layer_id = _check_path_found(_new_hashes.to(path_device), bfs_layers_hashes)
            if profile is not None:
                _cuda_sync()
                profile.check += time.time() - t1
            if bfs_layer_id != -1:
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

            # 5. Non-backtracking: single get_mask_to_remove_seen_hashes per slot
            # (same compact hash-set as search_iterated Task 1.6).
            if history_depth > 0:
                if profile is not None:
                    _cuda_sync()
                    t1 = time.time()
                _nb_mask = torch.ones_like(_new_hashes, dtype=torch.bool)
                for _slot in nonbacktrack_hashes:
                    _nb_mask &= _slot.get_mask_to_remove_seen_hashes(_new_hashes)
                # Add this step's hashes to the current slot (sorted — from step 3).
                nonbacktrack_hashes[i_cyclic_index_for_hash_storage].add_sorted_hashes(_new_hashes)
                # Apply mask unconditionally (Task 1.3 pattern).
                _new_states = _new_states[_nb_mask, :]
                _new_hashes = _new_hashes[_nb_mask]
                _gen_origin = _gen_origin[_nb_mask]
                if profile is not None:
                    _cuda_sync()
                    profile.isin += time.time() - t1

            if _new_hashes.shape[0] == 0:
                if verbose >= 1:
                    print(f"Cannot find new states at step {i_step}.")
                return BeamSearchResult(False, i_step, None, debug_scores, graph.definition)

            # 6. Per-generator topk with surplus redistribution.
            # Each generator gets up to beam_width_part slots (preserves fairness).
            # Surplus (generators with > part survivors) is redistributed to generators
            # with < part survivors, filled from the global pool by score.
            if profile is not None:
                _cuda_sync()
                t1 = time.time()
            if _new_states.shape[0] > beam_width:
                # Score ALL survivors once (ONE predictor call vs n_gens per-chunk in iterated).
                _all_scores = _predictor(graph.decode_states(_new_states))

                # Per-generator selection.
                _sel_states_list: list[torch.Tensor] = []
                _sel_hashes_list: list[torch.Tensor] = []
                _sel_global_idx_list: list[torch.Tensor] = []
                _slots_used = 0
                for _g in range(n_generators):
                    _g_mask = _gen_origin == _g
                    _g_global_idx = torch.where(_g_mask)[0]
                    _g_states = _new_states[_g_mask]
                    _g_hashes = _new_hashes[_g_mask]
                    _g_scores = _all_scores[_g_mask]
                    if _g_states.shape[0] > beam_width_part:
                        _vals, _idx = torch.topk(_g_scores, k=beam_width_part, largest=False, sorted=True)
                        _g_states = _g_states[_idx]
                        _g_hashes = _g_hashes[_idx]
                        # Map topk indices back to GLOBAL indices (fixes surplus bug:
                        # previously _g_global[:_keep] took hash-first, not topk-selected).
                        _g_global_idx = _g_global_idx[_idx]
                    _sel_states_list.append(_g_states)
                    _sel_hashes_list.append(_g_hashes)
                    _sel_global_idx_list.append(_g_global_idx)
                    _slots_used += _g_states.shape[0]

                # Surplus redistribution: if total selected < beam_width, fill remaining
                # slots from the global pool (all survivors not yet selected), by score.
                if _slots_used < beam_width:
                    _selected_mask = torch.zeros(_new_states.shape[0], dtype=torch.bool, device=graph.device)
                    for _g_idx in _sel_global_idx_list:
                        _selected_mask[_g_idx] = True
                    _remaining_idx = torch.where(~_selected_mask)[0]
                    _remaining_scores = _all_scores[_remaining_idx]
                    _n_fill = min(beam_width - _slots_used, _remaining_scores.shape[0])
                    if _n_fill > 0:
                        _fill_vals, _fill_idx = torch.topk(_remaining_scores, k=_n_fill, largest=False, sorted=True)
                        _fill_global_idx = _remaining_idx[_fill_idx]
                        _sel_states_list.append(_new_states[_fill_global_idx])
                        _sel_hashes_list.append(_new_hashes[_fill_global_idx])

                beam_states = torch.cat(_sel_states_list, dim=0)[:beam_width]
                beam_hashes = torch.cat(_sel_hashes_list, dim=0)[:beam_width]

                best_score = float(torch.min(_all_scores))
                debug_scores[i_step] = best_score
                if verbose >= 2:
                    print(f"Step {i_step}, best score: {best_score:.2f}.")
            else:
                beam_states = _new_states
                beam_hashes = _new_hashes
                if verbose >= 2:
                    print(f"Step {i_step}, not scored cause beam_width is big enough.")

            if profile is not None:
                _cuda_sync()
                profile.predict += time.time() - t1

            # beam_hashes is NOT sorted after per-generator topk (generators are
            # concatenated in gen order, each sub-block sorted by score). For path
            # restoration this is fine (uses isin membership, not order). For the
            # next step's nonbacktrack add_sorted_hashes, we need sorted — but that
            # add happens in step 5 which re-sorts via the dedup inline. So no
            # explicit re-sort needed here.
            if return_path:
                restore_path_hashes.append(beam_hashes.to(path_device))

            if memory_cleanup:
                graph.free_memory()

            if verbose >= 10 and (i_step - 1) % 10 == 0:
                print(f"Step {i_step}, beam size: {beam_states.shape[0]}.")

            if profile is not None:
                _cuda_sync()
                print(profile.format_line(i_step, t0))

        if verbose >= 1:
            print(f"Path not found after {max_steps} steps.")

        return BeamSearchResult(False, max_steps, None, debug_scores, graph.definition)
