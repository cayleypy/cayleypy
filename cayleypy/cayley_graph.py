import gc
import math
import warnings
from functools import cached_property
from typing import Optional, Sequence, Union

import torch

from .algo.beam_search import BeamSearchAlgorithm
from .algo.bfs_algo import BfsAlgorithm
from .algo.bfs_distributed import BfsDistributed
from .algo.random_walks import RandomWalksGenerator
from .bfs_result import BfsResult
from .cayley_graph_def import AnyStateType, CayleyGraphDef, GeneratorType
from .hasher import StateHasher
from .string_encoder import StringEncoder
from .torch_utils import isin_via_searchsorted


class CayleyGraph:
    """Represents a Schreier coset graph for some group."""

    def __init__(
        self,
        definition: CayleyGraphDef,
        *,
        device: str = "auto",
        random_seed: Optional[int] = None,
        bit_encoding_width: Union[Optional[int], str] = "auto",
        verbose: int = 0,
        batch_size: int = 2**20,
        hash_chunk_size: int = 2**25,
        memory_limit_gb: float = 16,
        num_gpus: Optional[int] = None,
        specific_devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
        _hasher: Optional[StateHasher] = None,
        **unused_kwargs,
    ):
        """Initializes CayleyGraph."""
        self.definition = definition
        self.verbose = verbose
        self.batch_size = batch_size
        self.memory_limit_bytes = int(memory_limit_gb * (2**30))
        self.bit_encoding_width = bit_encoding_width
        self._device_arg = device
        self._num_gpus_arg = num_gpus
        self._specific_devices_arg = list(specific_devices) if specific_devices is not None else None

        self.device, self.gpu_devices = self._resolve_devices(device, num_gpus, specific_devices)
        self.num_gpus = len(self.gpu_devices)
        if verbose > 0:
            print(f"Using device: {self.device}.")

        self.central_state = torch.as_tensor(definition.central_state, device=self.device, dtype=torch.int64)
        self.encoded_state_size: int = self.definition.state_size
        self.string_encoder: Optional[StringEncoder] = None
        self._device_permutations: dict[torch.device, torch.Tensor] = {}

        if definition.is_permutation_group():
            self.permutations_torch = torch.tensor(
                definition.generators_permutations, dtype=torch.int64, device=self.device
            )
            if bit_encoding_width == "auto":
                bit_encoding_width = int(math.ceil(math.log2(int(self.central_state.max()) + 1)))
            if bit_encoding_width is not None:
                self.string_encoder = StringEncoder(code_width=int(bit_encoding_width), n=self.definition.state_size)
                self.encoded_generators = [
                    self.string_encoder.implement_permutation(perm) for perm in definition.generators_permutations
                ]
                self.encoded_state_size = self.string_encoder.encoded_length

        if _hasher is not None:
            self.hasher = _hasher
        else:
            self.hasher = StateHasher(self, random_seed, chunk_size=hash_chunk_size)
        self.central_state_hash = self.hasher.make_hashes(self.encode_states(self.central_state))

    @staticmethod
    def _normalize_cuda_device(device: Union[int, str, torch.device]) -> torch.device:
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        normalized = torch.device(device)
        if normalized.type == "cuda" and normalized.index is None:
            return torch.device("cuda:0")
        return normalized

    def _resolve_devices(
        self,
        device: str,
        num_gpus: Optional[int],
        specific_devices: Optional[Sequence[Union[int, str, torch.device]]],
    ) -> tuple[torch.device, list[torch.device]]:
        if specific_devices is not None:
            resolved = [self._normalize_cuda_device(dev) for dev in specific_devices]
            if not resolved:
                raise ValueError("specific_devices must not be empty.")
            if not torch.cuda.is_available():
                raise ValueError("specific_devices requires CUDA, but CUDA is not available.")
            available = torch.cuda.device_count()
            for dev in resolved:
                if dev.type != "cuda":
                    raise ValueError("specific_devices must contain only CUDA devices.")
                if dev.index is None or dev.index >= available:
                    raise ValueError(f"CUDA device {dev} is not available.")
            return resolved[0], resolved

        if device == "gpu":
            device = "cuda"
        if device not in ["auto", "cpu", "cuda"]:
            raise ValueError("device must be one of 'auto', 'cpu', 'cuda', or 'gpu'.")

        if device == "cpu":
            if num_gpus not in [None, 0, 1]:
                raise ValueError("device='cpu' only supports num_gpus=None, 0, or 1.")
            if num_gpus == 1:
                warnings.warn("num_gpus=1 was provided with device='cpu'; using the CPU single-device path.")
            return torch.device("cpu"), []

        if device == "auto":
            if num_gpus == 0:
                return torch.device("cpu"), []
            if not torch.cuda.is_available():
                if num_gpus not in [None, 1]:
                    raise ValueError("num_gpus was requested, but CUDA is not available.")
                return torch.device("cpu"), []
            device = "cuda"

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but CUDA is not available.")

        available = torch.cuda.device_count()
        if num_gpus is None:
            resolved_num_gpus = available
        else:
            if num_gpus <= 0:
                raise ValueError("num_gpus must be positive when CUDA is selected.")
            if num_gpus > available:
                raise ValueError(f"Requested {num_gpus} GPUs, but only {available} are available.")
            resolved_num_gpus = num_gpus

        gpu_devices = [torch.device(f"cuda:{i}") for i in range(resolved_num_gpus)]
        return gpu_devices[0], gpu_devices

    def _get_permutations_for_device(self, device: torch.device) -> torch.Tensor:
        if device == self.permutations_torch.device:
            return self.permutations_torch
        if device not in self._device_permutations:
            self._device_permutations[device] = self.permutations_torch.to(device)
        return self._device_permutations[device]

    def get_unique_states(
        self, states: torch.Tensor, hashes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states` and sorts them by hash."""
        if self.hasher.is_identity:
            unique_hashes = torch.unique(states.reshape(-1), sorted=True)
            return unique_hashes.reshape((-1, 1)), unique_hashes
        if hashes is None:
            hashes = self.hasher.make_hashes(states)
        hashes_sorted, idx = torch.sort(hashes, stable=True)
        mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=hashes_sorted.device)
        if hashes_sorted.size(0) > 1:
            mask[1:] = hashes_sorted[1:] != hashes_sorted[:-1]
        unique_idx = idx[mask]
        return states[unique_idx], hashes[unique_idx]

    def encode_states(self, states: AnyStateType) -> torch.Tensor:
        """Converts states from human-readable to internal representation."""
        states = torch.as_tensor(states, device=self.device)
        states = states.reshape((-1, self.definition.state_size))
        if self.string_encoder is not None:
            return self.string_encoder.encode(states)
        return states

    def decode_states(self, states: torch.Tensor) -> torch.Tensor:
        """Converts states from internal to human-readable representation."""
        if self.definition.generators_type == GeneratorType.MATRIX:
            n, m = self.definition.decoded_state_shape
            return states.reshape((-1, n, m))
        if self.string_encoder is not None:
            return self.string_encoder.decode(states)
        return states

    def apply_generator_batched(self, i: int, src: torch.Tensor, dst: torch.Tensor):
        """Applies i-th generator to encoded states in `src`, writes output to `dst`."""
        states_num = src.shape[0]
        if self.definition.is_permutation_group():
            if self.string_encoder is not None:
                self.encoded_generators[i](src, dst)
            else:
                perms = self._get_permutations_for_device(src.device)
                move = perms[i].reshape((1, -1)).expand(states_num, -1)
                dst[:, :] = torch.gather(src, 1, move)
        else:
            assert self.definition.is_matrix_group()
            n, m = self.definition.decoded_state_shape
            mx = self.definition.generators_matrices[i]
            src = src.reshape((states_num, n, m))
            dst[:, :] = mx.apply_batch_torch(src).reshape((states_num, n * m))

    def apply_path(self, states: AnyStateType, generator_ids: list[int]) -> torch.Tensor:
        """Applies multiple generators to given state(s) in order."""
        states = self.encode_states(states)
        for gen_id in generator_ids:
            assert 0 <= gen_id < self.definition.n_generators
            new_states = torch.zeros_like(states)
            self.apply_generator_batched(gen_id, states, new_states)
            states = new_states
        return self.decode_states(states)

    def validate_path(self, start_state: AnyStateType, path: list[int]):
        """Checks that `path` indeed is path from `start_state` to central state."""
        path_result = self.apply_path(start_state, path).reshape(-1)
        assert torch.equal(path_result, self.central_state)

    def get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates all neighbors of `states` (in internal representation)."""
        states_num = states.shape[0]
        neighbors = torch.zeros(
            (states_num * self.definition.n_generators, states.shape[1]),
            dtype=torch.int64,
            device=states.device,
        )
        for i in range(self.definition.n_generators):
            dst = neighbors[i * states_num : (i + 1) * states_num, :]
            self.apply_generator_batched(i, states, dst)
        return neighbors

    def get_neighbors_decoded(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates neighbors in decoded (external) representation."""
        return self.decode_states(self.get_neighbors(self.encode_states(states)))

    def bfs(self, **kwargs) -> BfsResult:
        """Runs bread-first search (BFS) algorithm from given `start_states`."""
        if self.num_gpus > 1:
            return BfsDistributed.bfs(self, **kwargs)
        return BfsAlgorithm.bfs(self, **kwargs)

    def random_walks(self, **kwargs):
        """Generates random walks on this graph."""
        return RandomWalksGenerator(self).generate(**kwargs)

    def beam_search(self, **kwargs):
        """Tries to find a path from `start_state` to central state using Beam Search algorithm."""
        return BeamSearchAlgorithm(self).search(**kwargs)

    def restore_path(self, hashes: list[torch.Tensor], to_state: AnyStateType) -> list[int]:
        """Restores path from layers hashes."""
        inv_graph = self.with_inverted_generators
        path = []
        cur_state = self.decode_states(self.encode_states(to_state))

        for i in range(len(hashes) - 1, -1, -1):
            candidates = inv_graph.get_neighbors_decoded(cur_state)
            candidates_hashes = self.hasher.make_hashes(self.encode_states(candidates))
            mask = torch.isin(candidates_hashes, hashes[i])
            assert torch.any(mask), "Not found any neighbor on previous layer."
            gen_id = int(mask.nonzero()[0].item())
            path.append(gen_id)
            cur_state = candidates[gen_id : gen_id + 1, :]
        return path[::-1]

    def find_path_to(self, end_state: AnyStateType, bfs_result: BfsResult) -> Optional[list[int]]:
        """Finds path from central_state to end_state using pre-computed BfsResult."""
        assert bfs_result.graph == self.definition
        end_state_hash = self.hasher.make_hashes(self.encode_states(end_state))
        bfs_result.check_has_layer_hashes()
        layers_hashes = bfs_result.layers_hashes
        for i, bfs_layer in enumerate(layers_hashes):
            if bool(isin_via_searchsorted(end_state_hash, bfs_layer)):
                return self.restore_path(layers_hashes[:i], end_state)
        return None

    def find_path_from(self, start_state: AnyStateType, bfs_result: BfsResult) -> Optional[list[int]]:
        """Finds path from start_state to central_state using pre-computed BfsResult."""
        assert self.definition.generators_inverse_closed
        path_to = self.find_path_to(start_state, bfs_result)
        if path_to is None:
            return None
        return self.definition.revert_path(path_to)

    def to_networkx_graph(self):
        return self.bfs(
            max_layer_size_to_store=10**18, return_all_edges=True, return_all_hashes=True
        ).to_networkx_graph()

    def free_memory(self):
        if self.verbose >= 1:
            print("Freeing memory...")
        gc.collect()
        for dev in self.gpu_devices or [self.device]:
            if dev.type == "cuda":
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()
        gc.collect()

    @property
    def generators(self):
        """Generators of this Cayley graph."""
        return self.definition.generators

    @cached_property
    def with_inverted_generators(self):
        """Returns copy of this graph with inverted generators."""
        return self.modified_copy(self.definition.with_inverted_generators())

    def modified_copy(self, new_def: CayleyGraphDef) -> "CayleyGraph":
        """Makes a copy of this graph with different definition but other parameters unchanged."""
        ans = CayleyGraph(
            new_def,
            device=self._device_arg,
            _hasher=self.hasher,
            bit_encoding_width=self.bit_encoding_width,
            num_gpus=self._num_gpus_arg,
            specific_devices=self._specific_devices_arg,
        )
        ans.hasher = self.hasher
        ans.string_encoder = self.string_encoder
        return ans
