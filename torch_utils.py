import torch


def isin_via_searchsorted(elements: torch.Tensor, test_elements_sorted: torch.Tensor):
    """Equivalent to torch.isin but faster."""
    if len(test_elements_sorted) == 0:
        return torch.zeros_like(elements, dtype=torch.bool)
    ts = torch.searchsorted(test_elements_sorted, elements)
    ts[ts >= len(test_elements_sorted)] = len(test_elements_sorted) - 1
    return test_elements_sorted[ts] == elements


class TorchHashSet:
    """A set of int64 numbers, backed by one or more sorted tensors."""

    def __init__(self):
        self.data = []

    def add_sorted_hashes(self, sorted_numbers: torch.Tensor):
        """IMPORTANT: Assumes that new numbers are sorted and do not appear in the set before."""
        self.data.append(sorted_numbers)
        if len(self.data) >= 10:
            new_data, _ = torch.hstack(self.data).sort()
            self.data = [new_data]

    def get_mask_to_remove_seen_hashes(self, x: torch.Tensor) -> torch.Tensor:
        mask = ~isin_via_searchsorted(x, self.data[0])
        for i in range(1, len(self.data)):
            mask &= ~isin_via_searchsorted(x, self.data[i])
        return mask
