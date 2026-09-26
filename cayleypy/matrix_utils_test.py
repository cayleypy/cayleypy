from itertools import product
from math import gcd
import random

import numpy as np
import pytest

from .matrix_utils import inverse_integer, inverse_mod


def _check_integer_inverse(A):
    A = np.asarray(A, dtype=object)
    B = np.asarray(inverse_integer(A), dtype=object)
    identity = np.eye(len(A), dtype=object)

    assert np.array_equal(A @ B, identity)
    assert np.array_equal(B @ A, identity)


def _random_unimodular_matrix(n, rng):
    A = np.eye(n, dtype=object)

    for _ in range(5 * n):
        i, j = rng.sample(range(n), 2)
        operation = rng.randrange(3)

        if operation == 0:
            A[[i, j]] = A[[j, i]]
        elif operation == 1:
            c = rng.choice([-3, -2, -1, 1, 2, 3])
            A[i] += c * A[j]
        else:
            A[i] = -A[i]

    return A


def test_inverse_integer_known_matrices():
    assert inverse_integer([[1]]) == [[1]]
    assert inverse_integer([[-1]]) == [[-1]]

    assert inverse_integer(
        [[2, 3], [3, 5]]
    ) == [
        [5, -3],
        [-3, 2],
    ]

    assert inverse_integer(
        [[1, 2, 3], [0, 1, 4], [0, 0, 1]]
    ) == [
        [1, -2, 5],
        [0, 1, -4],
        [0, 0, 1],
    ]


def test_inverse_integer_without_unit_pivots():
    A = [
        [2, 3, 0, 0],
        [3, 5, 0, 0],
        [0, 0, 2, 3],
        [0, 0, 3, 5],
    ]
    _check_integer_inverse(A)


def test_inverse_integer_large_entries():
    # This unimodular example is already inaccurate with np.linalg.inv.
    A = [
        [17711, 10946],
        [10946, 6765],
    ]
    assert inverse_integer(A) == [
        [-6765, 10946],
        [10946, -17711],
    ]


def test_inverse_integer_2x2_exhaustive():
    for values in product(range(-2, 3), repeat=4):
        A = [list(values[:2]), list(values[2:])]
        determinant = values[0] * values[3] - values[1] * values[2]

        if abs(determinant) == 1:
            _check_integer_inverse(A)
        else:
            with pytest.raises(ValueError):
                inverse_integer(A)


def test_inverse_integer_generated_unimodular_matrices():
    rng = random.Random(12345)

    for n in range(2, 8):
        for _ in range(20):
            _check_integer_inverse(
                _random_unimodular_matrix(n, rng)
            )


def test_inverse_integer_numpy_input_and_no_mutation():
    A = np.array([[2, 3], [3, 5]], dtype=np.int64)
    A_before = A.copy()

    assert inverse_integer(A) == [[5, -3], [-3, 2]]
    assert np.array_equal(A, A_before)


def test_inverse_integer_noninvertible():
    matrices = [
        [[0]],
        [[2]],                         # Invertible over Q, not over Z.
        [[1, 2], [2, 4]],              # Singular.
        [[2, 1], [0, 1]],              # Determinant 2.
        [[1, 0, 0], [0, -1, 0], [0, 0, 3]],
    ]

    for A in matrices:
        with pytest.raises(ValueError):
            inverse_integer(A)


def test_inverse_integer_invalid_shape():
    for A in [[], [[1, 2]], [[1], [2]], [[1, 0], [0, 1, 0]]]:
        with pytest.raises(ValueError):
            inverse_integer(A)


def _check_inverse(A, m):
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(inverse_mod(A, m), dtype=np.int64)
    identity = np.eye(len(A), dtype=np.int64)

    assert np.array_equal(A @ B % m, identity)
    assert np.array_equal(B @ A % m, identity)
    assert np.all((0 <= B) & (B < m))


def _random_invertible_matrix(n, m, rng):
    A = np.eye(n, dtype=np.int64)
    units = [x for x in range(1, m) if gcd(x, m) == 1]

    for _ in range(5 * n):
        i, j = rng.sample(range(n), 2)
        operation = rng.randrange(3)
        if operation == 0:
            A[[i, j]] = A[[j, i]]
        elif operation == 1:
            c = rng.randrange(m)
            A[i] = (A[i] + c * A[j]) % m
        else:
            c = rng.choice(units)
            A[i] = c * A[i] % m

    return A


def test_inverse_mod_1x1():
    assert inverse_mod([[1]], 2) == [[1]]
    assert inverse_mod([[5]], 12) == [[5]]
    assert inverse_mod([[7]], 15) == [[13]]


def test_inverse_mod_2x2():
    assert inverse_mod([[1, 2], [3, 4]], 5) == [[3, 1], [4, 2]]
    assert inverse_mod([[2, 3], [3, 2]], 6) == [[2, 3], [3, 2]]
    assert inverse_mod([[-1, 2], [3, 6]], 5) == [[2, 1], [4, 3]]


def test_inverse_mod_2x2_exhaustive():
    for m in range(2, 8):
        for values in product(range(m), repeat=4):
            A = [list(values[:2]), list(values[2:])]
            determinant = values[0] * values[3] - values[1] * values[2]
            if gcd(determinant, m) == 1:
                _check_inverse(A, m)
            else:
                with pytest.raises(ValueError):
                    inverse_mod(A, m)


def test_inverse_mod_3x3():
    A = [[2, 1, 3], [1, 1, 1], [4, 2, 1]]
    assert inverse_mod(A, 7) == [[3, 6, 6], [5, 2, 4], [6, 0, 4]]

    _check_inverse([[1, 2, 3], [0, 1, 4], [0, 0, 1]], 12)
    _check_inverse([[2, 5, 1], [1, 3, 2], [4, 0, 3]], 11)


def test_inverse_mod_generic_without_unit_pivots():
    A = [
        [2, 3, 0, 0],
        [3, 2, 0, 0],
        [0, 0, 2, 3],
        [0, 0, 3, 2],
    ]
    assert inverse_mod(A, 6) == A


def test_inverse_mod_generated_invertible_matrices():
    rng = random.Random(12345)
    for n in range(2, 8):
        for m in [2, 4, 5, 6, 8, 9, 10, 12, 15]:
            for _ in range(10):
                A = _random_invertible_matrix(n, m, rng)
                _check_inverse(A, m)


def test_inverse_mod_numpy_input_and_no_mutation():
    A = np.array([[1, 2], [3, 4]], dtype=np.int64)
    A_before = A.copy()
    assert inverse_mod(A, 5) == [[3, 1], [4, 2]]
    assert np.array_equal(A, A_before)


def test_inverse_mod_noninvertible():
    matrices = [
        ([[1, 2], [2, 4]], 5),
        ([[1, 0, 0], [0, 2, 0], [0, 0, 1]], 6),
        ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]], 6),
    ]
    for A, m in matrices:
        with pytest.raises(ValueError):
            inverse_mod(A, m)


def test_inverse_mod_invalid_shape():
    for A in [[], [[1, 2]], [[1], [2]], [[1, 0], [0, 1, 0]]]:
        with pytest.raises(ValueError):
            inverse_mod(A, 5)

    with pytest.raises(ValueError):
        inverse_mod([[1]], 1)
