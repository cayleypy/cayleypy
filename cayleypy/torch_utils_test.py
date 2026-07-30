"""Tests for ``cayleypy.torch_utils``.

This module sits on the beam_search hot path (``isin_via_searchsorted`` is called
inside ``_check_path_found`` and ``_remove_seen_states``; ``TorchHashSet`` is used
by ``random_walks_bfs``). Full coverage is cheap given the module is only 30 lines.
"""

import torch

from cayleypy.torch_utils import TorchHashSet, isin_via_searchsorted


def _torch_isin(elements, test_elements):
    """Reference implementation using torch.isin (assumes unsorted test_elements)."""
    return torch.isin(elements, test_elements)


# =============================================================================
# isin_via_searchsorted
# =============================================================================


def test_isin_empty_test_elements():
    """Empty ``test_elements_sorted`` -> all False."""
    elements = torch.tensor([3, 1, 4, 1, 5])
    result = isin_via_searchsorted(elements, torch.tensor([], dtype=torch.int64))
    assert torch.equal(result, torch.tensor([False, False, False, False, False]))


def test_isin_empty_elements():
    """Empty ``elements`` -> empty result."""
    result = isin_via_searchsorted(torch.tensor([], dtype=torch.int64), torch.tensor([1, 2, 3]))
    assert len(result) == 0


def test_isin_sorted_test_elements():
    """Standard case: sorted test_elements, elements in arbitrary order."""
    test_elements_sorted = torch.tensor([1, 3, 5, 7, 9])
    elements = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
    result = isin_via_searchsorted(elements, test_elements_sorted)
    expected = _torch_isin(elements, test_elements_sorted)
    assert torch.equal(result, expected)


def test_isin_with_duplicates():
    """Duplicates in elements are handled correctly."""
    test_elements_sorted = torch.tensor([1, 2, 3])
    elements = torch.tensor([1, 1, 1, 2, 2, 3, 3, 3, 3])
    result = isin_via_searchsorted(elements, test_elements_sorted)
    assert torch.equal(result, torch.ones(9, dtype=torch.bool))


def test_isin_large_tensors_vs_torch_isin():
    """Equivalence with torch.isin on large random tensors."""
    test_elements_sorted, _ = torch.sort(torch.randint(0, 10_000, (5_000,), dtype=torch.int64))
    elements = torch.randint(0, 10_000, (20_000,), dtype=torch.int64)
    result = isin_via_searchsorted(elements, test_elements_sorted)
    expected = _torch_isin(elements, test_elements_sorted)
    assert torch.equal(result, expected)


def test_isin_single_element():
    """Single-element test_elements."""
    test_elements_sorted = torch.tensor([42])
    elements = torch.tensor([1, 42, 3, 42])
    result = isin_via_searchsorted(elements, test_elements_sorted)
    assert torch.equal(result, torch.tensor([False, True, False, True]))


def test_isin_clamps_out_of_range_indices():
    """Elements larger than the max test element are clamped to the last index (torch_utils.py:18).

    searchsorted returns len(test_elements_sorted) for elements exceeding the max.
    The clamp to len-1 makes the subsequent ``== elements`` check return False
    (the clamped index points at the max, which the element does not equal).
    Covers the clamp branch explicitly (otherwise only exercised indirectly).
    """
    test_elements_sorted = torch.tensor([10, 20, 30])
    # 99 and 100 exceed the max (30) -> searchsorted returns 3 (out of range).
    elements = torch.tensor([10, 99, 20, 100, 30, -5])
    result = isin_via_searchsorted(elements, test_elements_sorted)
    assert torch.equal(result, torch.tensor([True, False, True, False, True, False]))


# =============================================================================
# TorchHashSet
# =============================================================================


def test_torch_hash_set_add_and_mask():
    """Basic add + get_mask_to_remove_seen_hashes."""
    hs = TorchHashSet()
    sorted_hashes = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
    hs.add_sorted_hashes(sorted_hashes)
    # Elements 3 and 7 are in the set -> mask should be False for them.
    elements = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    mask = hs.get_mask_to_remove_seen_hashes(elements)
    assert torch.equal(mask, torch.tensor([False, True, False, True, False, True, False]))


def test_torch_hash_set_multiple_adds():
    """Adding multiple sorted batches works across shards."""
    hs = TorchHashSet()
    hs.add_sorted_hashes(torch.tensor([1, 3, 5], dtype=torch.int64))
    hs.add_sorted_hashes(torch.tensor([7, 9], dtype=torch.int64))
    elements = torch.tensor([1, 2, 3, 7, 8, 9], dtype=torch.int64)
    mask = hs.get_mask_to_remove_seen_hashes(elements)
    assert torch.equal(mask, torch.tensor([False, True, False, False, True, False]))


def test_torch_hash_set_coalescing_at_10_shards():
    """When ``len(data) >= 10``, shards are coalesced into one sorted tensor.

    Covers torch_utils.py:22-24.
    """
    hs = TorchHashSet()
    for i in range(10):
        hs.add_sorted_hashes(torch.tensor([i * 2, i * 2 + 1], dtype=torch.int64))
    # After 10 adds, data should have been coalesced into a single shard.
    assert len(hs.data) == 1
    # All 20 hashes should be present in the coalesced shard.
    elements = torch.tensor(list(range(20)), dtype=torch.int64)
    mask = hs.get_mask_to_remove_seen_hashes(elements)
    assert torch.equal(mask, torch.zeros(20, dtype=torch.bool))


def test_torch_hash_set_empty():
    """Empty set: all elements pass (none are seen)."""
    hs = TorchHashSet()
    elements = torch.tensor([1, 2, 3], dtype=torch.int64)
    mask = hs.get_mask_to_remove_seen_hashes(elements)
    assert torch.equal(mask, torch.tensor([True, True, True]))
    assert len(hs) == 0
    merged = hs.get_merged_sorted()
    assert len(merged) == 0


def test_torch_hash_set_empty_with_device():
    """B4: get_merged_sorted with explicit device returns empty tensor on that device."""
    hs = TorchHashSet()
    device = torch.device("cpu")
    merged = hs.get_merged_sorted(device=device)
    assert len(merged) == 0
    assert merged.device == device


def test_torch_hash_set_get_merged_sorted():
    """get_merged_sorted collapses shards into one sorted tensor."""
    hs = TorchHashSet()
    hs.add_sorted_hashes(torch.tensor([5, 7, 9], dtype=torch.int64))
    hs.add_sorted_hashes(torch.tensor([1, 3], dtype=torch.int64))
    assert len(hs.data) == 2
    merged = hs.get_merged_sorted()
    # After merge, only one shard remains, sorted.
    assert len(hs.data) == 1
    assert torch.equal(merged, torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64))
    # Subsequent queries are a single isin_via_searchsorted (no Python loop).
    mask = hs.get_mask_to_remove_seen_hashes(torch.tensor([1, 2, 9], dtype=torch.int64))
    assert torch.equal(mask, torch.tensor([False, True, False]))


def test_torch_hash_set_len():
    """__len__ sums shard sizes (including pre-merge)."""
    hs = TorchHashSet()
    assert len(hs) == 0
    hs.add_sorted_hashes(torch.tensor([1, 3], dtype=torch.int64))
    assert len(hs) == 2
    hs.add_sorted_hashes(torch.tensor([5], dtype=torch.int64))
    assert len(hs) == 3
    # After merge, len is preserved.
    hs.get_merged_sorted()
    assert len(hs) == 3


def test_torch_hash_set_add_empty_no_op():
    """Adding an empty tensor is a no-op (no empty shard appended)."""
    hs = TorchHashSet()
    hs.add_sorted_hashes(torch.tensor([], dtype=torch.int64))
    assert len(hs.data) == 0
    assert len(hs) == 0
    # A real add still works after the no-op.
    hs.add_sorted_hashes(torch.tensor([1, 2], dtype=torch.int64))
    assert len(hs.data) == 1
