import torch


def isin_via_searchsorted(elements: torch.Tensor, test_elements_sorted: torch.Tensor):
    """Equivalent to torch.isin but faster."""
    ts = torch.searchsorted(test_elements_sorted, elements)
    ts[ts >= len(test_elements_sorted)] = len(test_elements_sorted) - 1
    return (test_elements_sorted[ts] == elements)

def sort_and_unique(x: torch.Tensor) -> torch.Tensor:
    """Removes duplicates from `states`. May change order."""
    if x.size(0) <=1:
        return x
    x_sorted, idx = torch.sort(x, stable=True)    
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[1:] = (x_sorted[1:] != x_sorted[:-1])
    return x[idx[mask]]
    