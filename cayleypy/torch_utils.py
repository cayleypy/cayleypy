import torch


def isin_via_searchsorted(elements: torch.Tensor, test_elements_sorted: torch.Tensor):
    """Equivalent to torch.isin but faster.

    Requires `test_elements_sorted` to be sorted (uses `torch.searchsorted`).
    The sorted-order invariant is established by `CayleyGraph.get_unique_states`
    (see AGENTS.md §6 — "Hidden critical contract"). Breaking that invariant here
    silently breaks MITM path detection and deduplication.
    """
    if len(test_elements_sorted) == 0:
        return torch.zeros_like(elements, dtype=torch.bool)
    # Clamp out-of-range indices to the last valid index. searchsorted may return
    # len(...) for elements larger than the max; clamping (1 fused kernel) is cheaper
    # than the masked-assign it replaces (compare + indexed-put = 2 kernels) and is
    # semantically identical (the subsequent == elements check filters the clamped
    # rows back to False since they did not match any test element).
    ts = torch.searchsorted(test_elements_sorted, elements)
    ts = torch.clamp(ts, max=len(test_elements_sorted) - 1)
    return test_elements_sorted[ts] == elements


class TorchHashSet:
    """A set of int64 numbers, backed by one or more sorted tensors.

    Used by beam search for non-backtracking (history-depth slots) and for
    accumulator deduplication. Queries are backed by `isin_via_searchsorted`,
    which requires each shard to be individually sorted (the per-chunk hashes
    are sorted at the `beam_search.py` sort step — see AGENTS.md §6 invariant).

    Shards accumulate via `add_sorted_hashes` and are merged into a single
    sorted tensor once the shard count reaches `_MERGE_THRESHOLD` (avoids
    unbounded shard-list growth and the Python loop in
    `get_mask_to_remove_seen_hashes`).
    """

    _MERGE_THRESHOLD = 10

    def __init__(self):
        self.data: list[torch.Tensor] = []

    def add_sorted_hashes(self, sorted_numbers: torch.Tensor):
        """Append a sorted shard of hashes.

        IMPORTANT: `sorted_numbers` MUST be sorted AND must not contain any hash
        already present in the set. Both preconditions hold for per-chunk beam
        search hashes (each chunk is sorted, and chunks are deduplicated against
        the accumulator before being added here — see beam_search.py nonbacktrack
        block: the chunk is first deduped, THEN written to the history slot).
        """
        if len(sorted_numbers) > 0:
            self.data.append(sorted_numbers)
            if len(self.data) >= self._MERGE_THRESHOLD:
                merged, _ = torch.hstack(self.data).sort()
                self.data = [merged]

    def get_mask_to_remove_seen_hashes(self, x: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask: True where `x` is NOT in the set.

        Implementation note: when there are multiple shards, this loops over
        them in Python. Callers that query frequently should call
        `get_merged_sorted()` once and use `isin_via_searchsorted` directly to
        avoid the per-call Python loop. The merge threshold keeps the shard
        count bounded (≤ 10) so the loop is short.
        """
        if len(self.data) == 0:
            return torch.ones_like(x, dtype=torch.bool)
        mask = ~isin_via_searchsorted(x, self.data[0])
        for i in range(1, len(self.data)):
            mask &= ~isin_via_searchsorted(x, self.data[i])
        return mask

    def get_merged_sorted(self) -> torch.Tensor:
        """Merge all shards into one sorted tensor and return it.

        After this call, `self.data` holds exactly one tensor, so subsequent
        `get_mask_to_remove_seen_hashes` calls are a single
        `isin_via_searchsorted` (no Python loop). Use this when about to query
        the set many times (e.g. once per chunk per step in beam search).
        """
        if len(self.data) > 1:
            merged, _ = torch.hstack(self.data).sort()
            self.data = [merged]
        if len(self.data) == 0:
            return torch.empty(0, dtype=torch.int64, device=self._device())
        return self.data[0]

    def _device(self) -> torch.device:
        """Device of the first shard (CPU if the set is empty)."""
        return self.data[0].device if len(self.data) > 0 else torch.device("cpu")

    def __len__(self) -> int:
        return sum(len(shard) for shard in self.data)
