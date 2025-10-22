from math import factorial
import pytest
from .permutation_utils import permutations_with_cycle_lenghts, partition_to_permutation


@pytest.mark.unit
def test_permutations_with_cycle_lenghts_n_3():
    perms = permutations_with_cycle_lenghts(3, [3])
    assert len(perms) == 2
    assert all(sorted(p) == [0, 1, 2] for p in perms)
    assert perms == [[1, 2, 0], [2, 0, 1]]

    perms = permutations_with_cycle_lenghts(3, [2, 1])
    assert len(perms) == 3
    assert all(sorted(p) == [0, 1, 2] for p in perms)
    assert perms == [[0, 2, 1], [1, 0, 2], [2, 1, 0]]


@pytest.mark.unit
def test_permutations_with_cycle_lenghts_n_4():
    perms = permutations_with_cycle_lenghts(4, [4])
    assert len(perms) == 6
    assert all(sorted(p) == [0, 1, 2, 3] for p in perms)
    assert perms == [[1, 2, 3, 0], [1, 3, 0, 2], [2, 3, 1, 0], [2, 0, 3, 1], [3, 2, 0, 1], [3, 0, 1, 2]]

    # perms = permutations_with_cycle_lenghts(4, [3, 1])
    # assert len(perms) == 8
    # assert all(sorted(p) == [0, 1, 2, 3] for p in perms)
    # assert perms == [
    #     [0, 2, 3, 1],
    #     [0, 3, 1, 2],
    #     [1, 2, 0, 3],
    #     [2, 0, 1, 3],
    #     [1, 3, 2, 0],
    #     [3, 0, 2, 1],
    #     [2, 1, 3, 0],
    #     [3, 1, 0, 2],
    # ]

    perms = permutations_with_cycle_lenghts(4, [2, 2])
    assert len(perms) == 3
    assert all(sorted(p) == [0, 1, 2, 3] for p in perms)
    assert perms == [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]

    perms = permutations_with_cycle_lenghts(4, [2, 1, 1])
    assert len(perms) == 6
    assert all(sorted(p) == [0, 1, 2, 3] for p in perms)
    assert perms == [[0, 1, 3, 2], [0, 2, 1, 3], [0, 3, 2, 1], [1, 0, 2, 3], [2, 1, 0, 3], [3, 1, 2, 0]]


@pytest.mark.unit
def test_permutations_with_cycle_lenghts_n_8_sum():
    n = 8
    partitions = [
        [8],
        [7, 1],
        [6, 2],
        [6, 1, 1],
        [5, 3],
        [5, 2, 1],
        [5, 1, 1, 1],
        [4, 4],
        [4, 3, 1],
        [4, 2, 2],
        [4, 2, 1, 1],
        [4, 1, 1, 1, 1],
        [3, 3, 2],
        [3, 3, 1, 1],
        [3, 2, 2, 1],
        [3, 2, 1, 1, 1],
        [3, 1, 1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 1, 1],
        [2, 2, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1],
    ]
    counter = 0
    all_perms = set()
    for part in partitions:
        perms = permutations_with_cycle_lenghts(n, part)
        assert all(sorted(p) == list(range(n)) for p in perms)
        perms = permutations_with_cycle_lenghts(n, part)
        counter += len(perms)
        for p in perms:
            all_perms.add(tuple(p))
    assert counter + 1 == factorial(n)
    assert len(all_perms) + 1 == factorial(n)


@pytest.mark.unit
def test_partition_to_permutation():
    assert partition_to_permutation([2, 2]) == [1, 0, 3, 2]
    assert partition_to_permutation([5]) == [1, 2, 3, 4, 0]
    assert partition_to_permutation([1, 2, 3]) == [0, 2, 1, 4, 5, 3]
    assert partition_to_permutation([3, 2, 1]) == [1, 2, 0, 4, 3, 5]
    assert partition_to_permutation([3, 2, 2]) == [1, 2, 0, 4, 3, 6, 5]
    assert partition_to_permutation([2, 2, 2, 1, 1]) == [1, 0, 3, 2, 5, 4, 6, 7]

    for n in range(3, 11):
        perm = partition_to_permutation([n - 1, 1], True)
        assert sorted(perm) == list(range(n))
