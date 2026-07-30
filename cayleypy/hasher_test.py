"""Tests for ``cayleypy.hasher.StateHasher``.

The hasher sits on the beam_search hot path: ``make_hashes`` is called on every
beam expansion to deduplicate states and check the MITM neighborhood.
"""

import torch

from cayleypy.cayley_graph import CayleyGraph
from cayleypy.graphs_lib import MatrixGroups, PermutationGroups
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


def test_hasher_seed_zero_is_deterministic():
    """A2 regression: random_seed=0 must NOT be silently replaced by random seed."""
    graph1 = CayleyGraph(PermutationGroups.lrx(5), random_seed=0)
    graph2 = CayleyGraph(PermutationGroups.lrx(5), random_seed=0)
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
# Dual int32 CPU fast path (hasher.py:43-92)
# =============================================================================


def test_hasher_cpu_uses_dual_int32():
    """On CPU with a permutation group, the device-detection branch (hasher.py:43-59)
    selects the dual int32 path.

    The dual int32 path is the CPU-only fast path for permutation groups; GPU and
    matrix groups keep the int64 matmul path. This test guards against silently
    regressing to the int64 path on CPU for permutation groups.
    """
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=42)
    assert graph.device.type == "cpu"
    assert not graph.hasher.is_identity
    assert graph.hasher.make_hashes.__func__.__name__ == "_make_hashes_dual_int32"


def test_hasher_cpu_matrix_group_uses_int64_path():
    """Matrix groups on CPU must NOT use the dual int32 path (hasher.py:51 gate).

    The dual int32 path casts states to int32, which silently truncates int64
    matrix-group state values whose magnitude exceeds 2^31 (e.g. modulo==0 groups
    with large entries). Matrix groups must keep the int64 path to preserve dedup
    correctness. Regression guard for the AGENTS.md sec. 6 dedup invariant.
    """
    graph = CayleyGraph(MatrixGroups.heisenberg(modulo=0), device="cpu", random_seed=42)
    assert graph.device.type == "cpu"
    assert graph.definition.is_matrix_group()
    assert graph.hasher.make_hashes.__func__.__name__ != "_make_hashes_dual_int32"


def test_hasher_cpu_matrix_group_no_truncation_collision():
    """Distinct int64 matrix-group states differing only above bit 31 must hash
    differently on CPU (hasher.py:51 gate prevents int32 truncation).

    Before the permutation-group gate, ``states.to(torch.int32)`` dropped the high
    32 bits, so two states differing only in high bits collided — silently breaking
    ``get_unique_states`` dedup and MITM path detection. This guards that regression.
    """
    graph = CayleyGraph(MatrixGroups.heisenberg(modulo=0), device="cpu", random_seed=42)
    n = graph.definition.state_size
    s1 = torch.zeros((1, n), dtype=torch.int64)
    s2 = torch.zeros((1, n), dtype=torch.int64)
    s1[0, 0] = 2**40
    s2[0, 0] = 2**40 + 2**33  # differs only in bits above 31
    assert not torch.equal(s1, s2)
    h1 = graph.hasher.make_hashes(s1)
    h2 = graph.hasher.make_hashes(s2)
    assert not torch.equal(h1, h2)


def test_hasher_dual_int32_returns_int64():
    """The dual int32 path combines two int32 hashes into one int64 (hasher.py:74-92).

    int64 output is required by the sorted-hash invariant consumed by
    ``get_unique_states`` and ``isin_via_searchsorted`` (see AGENTS.md sec. 6).
    """
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=42)
    states = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=torch.int64)
    hashes = graph.hasher.make_hashes(states)
    assert hashes.dtype == torch.int64
    assert hashes.shape == (2,)


def test_hasher_dual_int32_no_collisions():
    """The dual int32 path keeps a 2^64 hash space: no collisions on 2^18 distinct states.

    Each int32 matmul wraps mod 2^32 (birthday bound 2^16), but the two hashes are
    independent so a collision requires both to collide (~2^-64 per pair). Safe for
    beams up to ~2^32 states. This guards the dedup invariant in ``get_unique_states``.
    """
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=42)
    states = torch.randint(0, 256, (2**18, 8), dtype=torch.int64)
    hashes = graph.hasher.make_hashes(states)
    n_unique_states = len(set(map(tuple, states.tolist())))
    assert hashes.unique().shape[0] == n_unique_states


def test_hasher_dual_int32_chunked_matches_full():
    """The dual int32 chunked path (hasher.py:85-92) matches the non-chunked path."""
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=42)
    hasher_chunked = StateHasher(graph, random_seed=42, chunk_size=4096)
    hasher_full = StateHasher(graph, random_seed=42, chunk_size=2**18)
    states = torch.randint(0, 256, (2**16, 8), dtype=torch.int64)
    hashes_chunked = hasher_chunked.make_hashes(states)
    hashes_full = hasher_full.make_hashes(states)
    assert torch.equal(hashes_chunked, hashes_full)


def test_hasher_dual_int32_deterministic_with_seed():
    """Same seed -> same dual int32 vectors -> same hashes (hasher.py:43-59)."""
    graph1 = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=42)
    graph2 = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=42)
    states = torch.randint(0, 256, (1000, 8), dtype=torch.int64)
    h1 = graph1.hasher.make_hashes(states)
    h2 = graph2.hasher.make_hashes(states)
    assert torch.equal(h1, h2)


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
