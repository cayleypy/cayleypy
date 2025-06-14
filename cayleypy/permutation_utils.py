"""Helper functions."""
from typing import Any, Sequence


def apply_permutation(p: Any, x: Sequence[Any]) -> list[Any]:
    return [x[p[i]] for i in range(len(p))]


def compose_permutations(p1: Sequence[int], p2: Sequence[int]) -> list[int]:
    """Returns p1∘p2."""
    return apply_permutation(p1, p2)


def inverse_permutation(p: Sequence[int]) -> list[int]:
    n = len(p)
    ans = [0] * n
    for i in range(n):
        ans[p[i]] = i
    return ans


def is_permutation(p: Any) -> bool:
    return sorted(list(p)) == list(range(len(p)))


def transposition(n: int, i1: int, i2: int) -> list[int]:
    """Returns permutation of n elements that is transposition (swap) of i1 and i2."""
    assert 0 <= i1 < n
    assert 0 <= i2 < n
    assert i1 != i2
    perm = list(range(n))
    perm[i1], perm[i2] = i2, i1
    return perm
