"""Tests for torchrun-aware beam search entry points."""

import os
import socket

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from ..cayley_graph import CayleyGraph
from ..graphs_lib import PermutationGroups
from .beam_search_multigpu import _compute_owner_pre_k, _compute_per_rank_beam, search_multigpu


def test_per_rank_beam_splits_remainder():
    assert [_compute_per_rank_beam(10, 4, rank) for rank in range(4)] == [3, 3, 2, 2]
    assert [_compute_per_rank_beam(2, 4, rank) for rank in range(4)] == [1, 1, 0, 0]


def test_owner_pre_k_uses_destination_rank_budget():
    assert _compute_owner_pre_k(10, 4, 0, 2) == 6
    assert _compute_owner_pre_k(10, 4, 2, 2) == 4
    assert _compute_owner_pre_k(2, 4, 3, 2) == 0


def test_search_multigpu_single_process_falls_back_to_advanced():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    result = search_multigpu(
        graph,
        start_state=[4, 1, 0, 2, 3],
        beam_width=20,
        max_steps=5,
    )
    assert result.path_found
    assert result.path_length == 2


def test_beam_mode_multigpu_single_process_falls_back_to_advanced():
    graph = CayleyGraph(PermutationGroups.lrx(6), device="cpu")
    start_state = np.array([5, 1, 0, 2, 3, 4])
    result = graph.beam_search(
        start_state=start_state,
        beam_mode="multigpu",
        beam_width=100,
        max_steps=5,
    )
    assert result.path_found
    assert result.path_length > 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _owner_partitioned_gloo_worker(rank: int, world_size: int, port: int) -> None:
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["USE_LIBUV"] = "0"

    result = search_multigpu(
        graph,
        start_state=[4, 1, 0, 2, 3],
        beam_width=20,
        max_steps=5,
        history_depth=1,
    )
    try:
        assert result.path_found
        assert result.path_length == 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(os.getenv("RUN_DISTRIBUTED_TESTS") != "1", reason="requires local multi-process gloo")
def test_owner_partitioned_gloo_spawn_smoke():
    mp.spawn(_owner_partitioned_gloo_worker, args=(2, _free_port()), nprocs=2, join=True)
