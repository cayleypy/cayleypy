"""Helper functions."""

from typing import Any, Sequence
from collections import Counter
import random
from itertools import combinations, permutations


def identity_perm(n: int) -> list[int]:
    return list(range(n))


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


def permutation_from_cycles(n: int, cycles: list[list[int]], offset: int = 0) -> list[int]:
    """Returns permutation of size n having given cycles."""
    perm = list(range(n))
    cycles_offsetted = [[x - offset for x in y] for y in cycles]

    for cycle in cycles_offsetted:
        for i in range(len(cycle)):
            assert 0 <= cycle[i] < n
            assert perm[cycle[i]] == cycle[i], "Cycles must not intersect."
            perm[cycle[i]] = cycle[(i + 1) % len(cycle)]
    return perm


def permutations_with_cycle_lenghts(n: int, cycle_lengths: list[int]) -> list[list[int]]:
    """
    Generates all 0-based permutations from S_n with given cycle lengths.

    @param: cycle_lengths, list[int]
        Contains positive integers which sum to n (order does not matter).

    Returns: list of permutations, each represented as a list of size n.

    Examples:
        n=3, cycle_lengths=[3]: returns all 3-cycles from S_3:
            [[1, 2, 0], [2, 0, 1]]

        n=4, cycle_lengths=[2, 1, 1]: returns all permutations from S_4 with one 2-cycle and two fixed points:
            [[0, 1, 3, 2], [0, 2, 1, 3], [0, 3, 2, 1], [1, 0, 2, 3], [2, 1, 0, 3], [3, 1, 2, 0]]

        n=5, cycle_lengths=[2, 3]: returns all permutations from S_5 with one 2-cycle and one 3-cycle:
            [[1, 0, 3, 4, 2], [1, 0, 4, 2, 3], [2, 3, 0, 4, 1], [2, 4, 0, 1, 3], [3, 2, 4, 0, 1],
             [3, 4, 1, 0, 2], [4, 2, 3, 1, 0], [4, 3, 1, 2, 0], [1, 2, 0, 4, 3], [2, 0, 1, 4, 3],
             [1, 3, 4, 0, 2], [3, 0, 4, 1, 2], [1, 4, 3, 2, 0], [4, 0, 3, 2, 1], [2, 4, 3, 0, 1],
             [3, 4, 0, 2, 1], [2, 3, 4, 1, 0], [4, 3, 0, 1, 2], [3, 2, 1, 4, 0], [4, 2, 1, 0, 3]]
    """
    assert n >= 1, "n must be at least 1"
    assert all(k >= 1 for k in cycle_lengths), "All cycle lengths must be positive"
    if sum(cycle_lengths) != n:
        raise ValueError("Sum of cycle lengths must equal n")

    lengths_counter = Counter(cycle_lengths)
    all_elems = set(range(n))
    result = []

    def backtrack(last_min, available, lengths_counter):
        # if no lengths left, we should have used all elements
        if not lengths_counter:
            if not available:
                yield []
            return

        # iterate available distinct cycle lengths in increasing order
        for k in sorted(lengths_counter):
            # choose combinations of available elements of size k
            for comb in combinations(sorted(available), k):
                m = comb[0]  # minimal element of this cycle
                if m <= last_min:
                    # ensure strict increase of minima -> canonical ordering
                    continue
                # fix rotation: keep m first, permute the rest
                rest = list(comb[1:])
                for order in permutations(rest):
                    cycle = (m,) + order
                    new_available = available - set(cycle)

                    # update multiset of lengths
                    new_lengths = lengths_counter.copy()
                    new_lengths[k] -= 1
                    if new_lengths[k] == 0:
                        del new_lengths[k]

                    # recurse
                    for tail_cycles in backtrack(m, new_available, new_lengths):
                        yield [cycle] + tail_cycles

    for cycles in backtrack(-1, all_elems, lengths_counter):
        arr = list(range(n))  # fixed points by default
        for cyc in cycles:
            size = len(cyc)
            for i in range(size):
                arr[cyc[i]] = cyc[(i + 1) % size]  # (a b c) maps a->b, b->c, c->a
        result.append(arr)
    return result


def partition_to_permutation(cycle_lengths: list[int], flag_random: bool = False) -> list[int]:
    """
    Returns a 0-based permutation from S_n with given cycle lengths.

    @param: cycle_lengths, list[int]
        Contains positive integers which sum to n and represent cycle lengths.

    @param: flag_random, bool
        If True, returns a random permutation with such cycle lenghts.
        If False, consequitively rearranges elements of the identity permutation
            to obtain a permutation with given cycle lengths.

    Examples:
        partition = [3, 2, 1], flag_random = False -> [1, 2, 0, 4, 3, 5]
        partition = [1, 2, 3], flag_random = False -> [0, 2, 1, 4, 5, 3]
        partition = [2, 2], flag_random = True -> [1, 0, 3, 2] or [2, 3, 0, 1] or [3, 2, 1, 0]
    """
    assert all(k >= 1 for k in cycle_lengths), "All cycle lengths must be positive"

    n = sum(cycle_lengths)
    elements = list(range(n))
    if flag_random:
        random.shuffle(elements)

    permutation = [0] * n
    idx = 0
    for size in cycle_lengths:
        cycle = elements[idx : idx + size]
        for i in range(size):
            permutation[cycle[i]] = cycle[(i + 1) % size]
        idx += size

    return permutation
