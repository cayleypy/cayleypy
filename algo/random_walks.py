"""Random walks generation for Cayley graphs."""

from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from ..torch_utils import TorchHashSet

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


class RandomWalksGenerator:
    """Generator for random walks on Cayley graphs.

    This class encapsulates the logic for generating random walks using different modes:

      * "classic": Simple random walks with independent steps.
      * "bfs": Breadth-first search based random walks with uniqueness constraints.
    """

    def __init__(self, graph: "CayleyGraph"):
        """Initialize the random walks generator.

        :param graph: The Cayley graph to generate walks on.
        """
        self.graph = graph

    def generate(
        self,
        *,
        width=5,
        length=10,
        mode="classic",
        start_state: Union[None, torch.Tensor, np.ndarray, list] = None,
        nbt_history_depth: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random walks on the graph.

        The following modes of random walk generation are supported:

          * "classic" - random walk is a path in this graph starting from `start_state`, where on each step the next
            edge is chosen randomly with equal probability. We generate `width` such random walks independently.
            The output will have exactly ``width*length`` states.
            i-th random walk can be extracted as: ``[x[i+j*width] for j in range(length)]``.
            ``y[i]`` is equal to number of random steps it took to get to state ``x[i]``.
            Note that in this mode a lot of states will have overestimated distance (meaning ``y[i]`` may be larger than
            the length of the shortest path from ``x[i]`` to `start_state`).
            The same state may appear multiple times with different distance in ``y``.
          * "bfs" - we perform Breadth First Search starting from ``start_state`` with one modification: if size of
            next layer is larger than ``width``, only ``width`` states (chosen randomly) will be kept.
            We also remove states from current layer if they appeared on some previous layer (so this also can be
            called "non-backtracking random walk").
            All states in the output are unique. ``y`` still can be overestimated, but it will be closer to the true
            distance than in "classic" mode. Size of output is ``<= width*length``.
            If ``width`` and ``length`` are large enough (``width`` at least as large as largest BFS layer, and
            ``length >= diameter``), this will return all states and true distances to the start state.
          * "nbt" - non-backtracking beam search random walks. This mode generates improved non-backtracking
            random walks that "mix even faster" - the number of random walk steps will be better related to actual
            distance on the graph. The procedure is similar to beam search, except there is no goal function. All
            trajectories know about each other and avoid visiting states visited by any trajectory. Additionally,
            1-neighbors of current states are also banned to improve mixing. The `nbt_history_depth` parameter
            controls how many previous levels to remember and ban.

        :param width: Number of random walks to generate.
        :param length: Length of each random walk.
        :param start_state: State from which to start random walk. Defaults to the central state.
        :param mode: Type of random walk (see above). Defaults to "classic".
        :param nbt_history_depth: For "nbt" mode, how many previous levels to remember and ban from revisiting.
        :return: Pair of tensors ``x, y``. ``x`` contains states. ``y[i]`` is the estimated distance from start state
          to state ``x[i]``.
        """
        start_state = self.graph.encode_states(start_state or self.graph.central_state)
        if mode == "classic":
            return self.random_walks_classic(width, length, start_state)
        elif mode == "bfs":
            return self.random_walks_bfs(width, length, start_state)
        elif mode == "nbt":
            return self.random_walks_nbt(width, length, start_state, nbt_history_depth)
        else:
            raise ValueError("Unknown mode:", mode)

    def random_walks_classic(
        self, width: int, length: int, start_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate classic random walks.

        :param width: Number of random walks to generate.
        :param length: Length of each random walk.
        :param start_state: Starting state for all walks.
        :return: Tuple of (states, distances).
        """
        # Allocate memory.
        graph = self.graph
        x_shape = (width * length, graph.encoded_state_size)
        x = torch.zeros(x_shape, device=graph.device, dtype=torch.int64)
        y = torch.zeros(width * length, device=graph.device, dtype=torch.int32)

        # First state in each walk is the start state.
        x[:width, :] = start_state.reshape((-1,))
        y[:width] = 0

        # Main loop.
        for i_step in range(1, length):
            y[i_step * width : (i_step + 1) * width] = i_step
            gen_idx = torch.randint(0, graph.definition.n_generators, (width,), device=graph.device)
            src = x[(i_step - 1) * width : i_step * width, :]
            dst = x[i_step * width : (i_step + 1) * width, :]
            for j in range(graph.definition.n_generators):
                # Go to next state for walks where we chose to use j-th generator on this step.
                mask = gen_idx == j
                prev_states = src[mask, :]
                next_states = torch.zeros_like(prev_states)
                graph.apply_generator_batched(j, prev_states, next_states)
                dst[mask, :] = next_states

        return graph.decode_states(x), y

    def random_walks_bfs(self, width: int, length: int, start_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate BFS-based random walks.

        :param width: Maximum number of states per layer.
        :param length: Maximum number of layers.
        :param start_state: Starting state for the BFS.
        :return: Tuple of (states, distances).
        """
        graph = self.graph
        x_hashes = TorchHashSet()
        x_hashes.add_sorted_hashes(graph.hasher.make_hashes(start_state))
        x = [start_state]
        y = [torch.full((1,), 0, device=graph.device, dtype=torch.int32)]

        for i_step in range(1, length):
            next_states = graph.get_neighbors(x[-1])
            next_states, next_states_hashes = graph.get_unique_states(next_states)
            mask = x_hashes.get_mask_to_remove_seen_hashes(next_states_hashes)
            next_states, next_states_hashes = next_states[mask], next_states_hashes[mask]
            layer_size = len(next_states)
            if layer_size == 0:
                break
            if layer_size > width:
                random_indices = torch.randperm(layer_size)[:width]
                layer_size = width
                next_states = next_states[random_indices]
                next_states_hashes = next_states_hashes[random_indices]
            x.append(next_states)
            x_hashes.add_sorted_hashes(next_states_hashes)
            y.append(torch.full((layer_size,), i_step, device=graph.device, dtype=torch.int32))
        return graph.decode_states(torch.vstack(x)), torch.hstack(y)

    def random_walks_nbt(
        self, width: int, length: int, start_state: torch.Tensor, history_depth: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate non-backtracking beam search random walks.

        This method generates improved non-backtracking random walks that "mix even faster" -
        the number of random walk steps will be better related to actual distance on the graph.
        The procedure is similar to beam search, except there is no goal function.
        All trajectories know about each other and avoid visiting states visited by any trajectory.

        :param width: Number of random walks to generate in parallel.
        :param length: Length of each random walk.
        :param start_state: Starting state for all walks (encoded).
        :param history_depth: How many previous levels to remember and ban from revisiting.
        :return: Tuple of (states, distances).
        """
        graph = self.graph

        # Initialize current states - duplicate start_state width times.
        array_current_states = start_state.view(1, -1).expand(width, -1).clone()

        # Allocate output arrays.
        states = torch.zeros(width * length, graph.encoded_state_size, device=graph.device, dtype=torch.int64)
        y = torch.zeros(width * length, device=graph.device, dtype=torch.int32)

        # Store initial states.
        states[:width, :] = array_current_states
        y[:width] = 0

        # Initialize hash storage for non-backtracking.
        if history_depth > 0:
            # Use graph's hasher for consistency.
            initial_hashes = graph.hasher.make_hashes(start_state)
            vec_hashes_current = initial_hashes.expand(width * graph.definition.n_generators, history_depth).clone()
            i_cyclic_index_for_hash_storage = 0

        i_step_corrected = 0
        for i_step in range(1, length):
            # 1. Create new states by applying all generators to all current states.
            array_new_states = graph.get_neighbors(array_current_states)
            # Ensure it's 2D: (n_states, state_size).
            if array_new_states.dim() == 1:
                array_new_states = array_new_states.unsqueeze(0)
            elif array_new_states.dim() > 2:
                array_new_states = array_new_states.flatten(end_dim=1)

            # 2. Non-backtracking: select states not seen before.
            if history_depth > 0:
                # Compute hashes of new states.
                vec_hashes_new = graph.hasher.make_hashes(array_new_states)

                # Select only states not seen before.
                mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
                mask_new_sum = mask_new.sum().item()

                if mask_new_sum >= width:
                    # Select only new states - not visited before.
                    array_new_states = array_new_states[mask_new, :]
                    i_step_corrected += 1
                else:
                    # Exceptional case: can't find enough new states.
                    # Take as many new states as possible and repeat if needed.
                    if mask_new_sum > 0:
                        repeat_factor = int(np.ceil(width / mask_new_sum))
                        array_new_states = array_new_states[mask_new, :].repeat(repeat_factor, 1)[:width, :]
                        i_step_corrected += 1
                    else:
                        # No new states found, stay in place.
                        array_new_states = array_current_states

            # 3. Select desired number of states randomly.
            perm = torch.randperm(array_new_states.size(0), device=graph.device)
            array_current_states = array_new_states[perm][:width]

            # 4. Store results.
            y[i_step * width : (i_step + 1) * width] = i_step_corrected
            states[i_step * width : (i_step + 1) * width, :] = array_current_states

            # 5. Update hash storage.
            if history_depth > 0:
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % history_depth
                vec_hashes_current[:, i_cyclic_index_for_hash_storage] = vec_hashes_new

        return graph.decode_states(states), y
