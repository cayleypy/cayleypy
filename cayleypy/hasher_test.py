"""Tests for ``cayleypy.hasher.StateHasher``.

The hasher sits on the beam_search hot path: ``make_hashes`` is called on every
beam expansion to deduplicate states and check the MITM neighborhood.
"""

import torch

from cayleypy.cayley_graph import CayleyGraph
from cayleypy.graphs_lib import PermutationGroups
from cayleypy.hasher import StateHasher, _splitmix64


# =============================================================================
# Identity hasher (state_size == 1)
# =============================================================================


def test_identity_hasher_single_element():
    """When state_size == 1, the hasher uses the identity function (hasher.py:30-33).

    With bit_encoding_width=2 on LRX(3), 3 elements * 2 bits = 6 bits, packed into
    1 int64 -> encoded_state_size == 1 -> identity hasher.
    """
    graph = CayleyGraph(PermutationGroups.lrx(3), bit_encoding_width=2)
    assert graph.encoded_state_size == 1
    assert graph.hasher.is_identity
    states = graph.encode_states(torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int64))
    hashes = graph.hasher.make_hashes(states)
    assert hashes.shape == (2,)
    assert torch.equal(hashes, states.reshape(-1))


def test_identity_hasher_is_identity_flag():
    """The ``is_identity`` flag is True when encoded_state_size == 1."""
    graph = CayleyGraph(PermutationGroups.lrx(3), bit_encoding_width=2)
    assert graph.encoded_state_size == 1
    assert graph.hasher.is_identity


# =============================================================================
# Non-identity hasher (state_size > 1)
# =============================================================================


def test_non_identity_hasher_flag():
    """The ``is_identity`` flag is False when state_size > 1."""
    graph = CayleyGraph(PermutationGroups.lrx(5))
    assert graph.encoded_state_size > 1
    assert not graph.hasher.is_identity


def test_hasher_deterministic_with_seed():
    """Same random_seed -> same hashes."""
    graph1 = CayleyGraph(PermutationGroups.lrx(5), random_seed=42)
    graph2 = CayleyGraph(PermutationGroups.lrx(5), random_seed=42)
    states = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=torch.int64)
    h1 = graph1.hasher.make_hashes(states)
    h2 = graph2.hasher.make_hashes(states)
    assert torch.equal(h1, h2)


def test_hasher_different_seeds_different_hashes():
    """Different random_seed -> (likely) different hashes."""
    graph1 = CayleyGraph(PermutationGroups.lrx(5), random_seed=1)
    graph2 = CayleyGraph(PermutationGroups.lrx(5), random_seed=2)
    states = torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.int64)
    h1 = graph1.hasher.make_hashes(states)
    h2 = graph2.hasher.make_hashes(states)
    assert not torch.equal(h1, h2)


def test_hasher_chunking_large_input():
    """Chunking path (hasher.py:57-63) is exercised for inputs > chunk_size."""
    graph = CayleyGraph(PermutationGroups.lrx(5), random_seed=42)
    hasher = StateHasher(graph, random_seed=42, chunk_size=4)
    # Input larger than chunk_size (4) triggers the chunked path.
    states = torch.randint(0, 5, (100, 5), dtype=torch.int64)
    hashes = hasher.make_hashes(states)
    assert hashes.shape == (100,)
    # Same result as non-chunked (chunk_size large enough).
    hasher_full = StateHasher(graph, random_seed=42, chunk_size=2**18)
    hashes_full = hasher_full.make_hashes(states)
    assert torch.equal(hashes, hashes_full)


# =============================================================================
# _splitmix64
# =============================================================================


def test_splitmix64_basic():
    """``_splitmix64`` is deterministic and returns int64 tensors."""
    x = torch.tensor([0, 1, 42, 2**62 - 1], dtype=torch.int64)
    h = _splitmix64(x)
    assert h.dtype == torch.int64
    assert h.shape == x.shape
    # Same input -> same output.
    assert torch.equal(_splitmix64(x), h)


def test_splitmix64_different_inputs_different_outputs():
    """Different inputs produce different outputs (with high probability)."""
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    h = _splitmix64(x)
    assert len(set(h.tolist())) == len(h)


# =============================================================================
# Splitmix64 hasher path (string_encoder set)
# =============================================================================


def test_splitmix64_hasher_path():
    """When ``string_encoder`` is set AND encoded_state_size > 1, splitmix64 hasher is used.

    LRX(5) with bit_encoding_width=13: 5*13=65 bits -> ceil(65/64)=2 int64s -> state_size=2.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5), bit_encoding_width=13)
    assert graph.string_encoder is not None
    assert graph.encoded_state_size == 2
    assert not graph.hasher.is_identity
    states = graph.encode_states(torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=torch.int64))
    hashes = graph.hasher.make_hashes(states)
    assert hashes.shape == (2,)
    assert hashes.dtype == torch.int64
