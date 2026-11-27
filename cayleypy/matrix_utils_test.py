from itertools import product
from math import gcd
import random
from typing import Union

import numpy as np
from numpy.typing import NDArray
import pytest

from .matrix_utils import inverse_integer, inverse_mod


def _check_integer_inverse(matrix: Union[list[list[int]], np.ndarray]) -> None:
    """Check both multiplication orders using exact integer arithmetic."""
    matrix = np.asarray(matrix, dtype=object)
    inverse = np.asarray(inverse_integer(matrix), dtype=object)
    identity = np.eye(len(matrix), dtype=object)

    assert np.array_equal(matrix @ inverse, identity)
    assert np.array_equal(inverse @ matrix, identity)


def _random_unimodular_matrix(n: int, rng: random.Random) -> NDArray[np.object_]:
    """Generate an integer matrix using invertible row operations."""
    matrix = np.eye(n, dtype=object)

    for _ in range(5 * n):
        i, j = rng.sample(range(n), 2)
        operation = rng.randrange(3)

        if operation == 0:
            matrix[[i, j]] = matrix[[j, i]]
        elif operation == 1:
            c = rng.choice([-3, -2, -1, 1, 2, 3])
            matrix[i] += c * matrix[j]
        else:
            matrix[i] = -matrix[i]

    return matrix


def test_inverse_integer_known_matrices() -> None:
    """Check exact inverses of small known unimodular matrices."""
    assert inverse_integer([[1]]) == [[1]]
    assert inverse_integer([[-1]]) == [[-1]]

    assert inverse_integer([[2, 3], [3, 5]]) == [
        [5, -3],
        [-3, 2],
    ]

    assert inverse_integer([[1, 2, 3], [0, 1, 4], [0, 0, 1]]) == [
        [1, -2, 5],
        [0, 1, -4],
        [0, 0, 1],
    ]


def test_inverse_integer_without_unit_pivots() -> None:
    """Exercise integer elimination when no initial pivot is a unit."""
    matrix = [
        [2, 3, 0, 0],
        [3, 5, 0, 0],
        [0, 0, 2, 3],
        [0, 0, 3, 5],
    ]
    _check_integer_inverse(matrix)


def test_inverse_integer_large_entries() -> None:
    """Check an inverse that floating-point inversion cannot recover exactly."""
    # This unimodular example is already inaccurate with np.linalg.inv.
    matrix = [
        [17711, 10946],
        [10946, 6765],
    ]
    assert inverse_integer(matrix) == [
        [-6765, 10946],
        [10946, -17711],
    ]


def test_inverse_integer_2x2_exhaustive() -> None:
    """Check invertibility against determinants for small 2x2 matrices."""
    for values in product(range(-2, 3), repeat=4):
        matrix = [list(values[:2]), list(values[2:])]
        determinant = values[0] * values[3] - values[1] * values[2]

        if abs(determinant) == 1:
            _check_integer_inverse(matrix)
        else:
            with pytest.raises(ValueError):
                inverse_integer(matrix)


def test_inverse_integer_generated_unimodular_matrices() -> None:
    """Verify exact inverses across generated matrix sizes."""
    rng = random.Random(12345)

    for n in range(2, 8):
        for _ in range(20):
            _check_integer_inverse(_random_unimodular_matrix(n, rng))


def test_inverse_integer_numpy_input_and_no_mutation() -> None:
    """Ensure integer inversion accepts NumPy arrays without changing them."""
    matrix = np.array([[2, 3], [3, 5]], dtype=np.int64)
    matrix_before = matrix.copy()

    assert inverse_integer(matrix) == [[5, -3], [-3, 2]]
    assert np.array_equal(matrix, matrix_before)


def test_inverse_integer_noninvertible() -> None:
    """Reject singular matrices and matrices with nonunit determinants."""
    matrices = [
        [[0]],
        [[2]],  # Invertible over Q, not over Z.
        [[1, 2], [2, 4]],  # Singular.
        [[2, 1], [0, 1]],  # Determinant 2.
        [[1, 0, 0], [0, -1, 0], [0, 0, 3]],
    ]

    for matrix in matrices:
        with pytest.raises(ValueError):
            inverse_integer(matrix)


def test_inverse_integer_invalid_shape() -> None:
    """Reject empty and nonsquare inputs to integer inversion."""
    for matrix in [[], [[1, 2]], [[1], [2]], [[1, 0], [0, 1, 0]]]:
        with pytest.raises(ValueError):
            inverse_integer(matrix)


def _check_inverse(matrix: Union[list[list[int]], np.ndarray], m: int) -> None:
    """Check both modular inverse identities and normalized entries."""
    matrix = np.asarray(matrix, dtype=np.int64)
    inverse = np.asarray(inverse_mod(matrix, m), dtype=np.int64)
    identity = np.eye(len(matrix), dtype=np.int64)

    assert np.array_equal(matrix @ inverse % m, identity)
    assert np.array_equal(inverse @ matrix % m, identity)
    assert np.all((0 <= inverse) & (inverse < m))


def _random_invertible_matrix(n: int, m: int, rng: random.Random) -> NDArray[np.int64]:
    """Generate an invertible modular matrix using elementary row operations."""
    matrix = np.eye(n, dtype=np.int64)
    units = [x for x in range(1, m) if gcd(x, m) == 1]

    for _ in range(5 * n):
        i, j = rng.sample(range(n), 2)
        operation = rng.randrange(3)
        if operation == 0:
            matrix[[i, j]] = matrix[[j, i]]
        elif operation == 1:
            c = rng.randrange(m)
            matrix[i] = (matrix[i] + c * matrix[j]) % m
        else:
            c = rng.choice(units)
            matrix[i] = c * matrix[i] % m

    return matrix


def test_inverse_mod_1x1() -> None:
    """Check scalar modular inverses."""
    assert inverse_mod([[1]], 2) == [[1]]
    assert inverse_mod([[5]], 12) == [[5]]
    assert inverse_mod([[7]], 15) == [[13]]


def test_inverse_mod_2x2() -> None:
    """Check known 2x2 inverses over prime and composite moduli."""
    assert inverse_mod([[1, 2], [3, 4]], 5) == [[3, 1], [4, 2]]
    assert inverse_mod([[2, 3], [3, 2]], 6) == [[2, 3], [3, 2]]
    assert inverse_mod([[-1, 2], [3, 6]], 5) == [[2, 1], [4, 3]]


def test_inverse_mod_2x2_exhaustive() -> None:
    """Check small modular matrices against the determinant criterion."""
    for m in range(2, 8):
        for values in product(range(m), repeat=4):
            matrix = [list(values[:2]), list(values[2:])]
            determinant = values[0] * values[3] - values[1] * values[2]
            if gcd(determinant, m) == 1:
                _check_inverse(matrix, m)
            else:
                with pytest.raises(ValueError):
                    inverse_mod(matrix, m)


def test_inverse_mod_3x3() -> None:
    """Check 3x3 adjugate inversion over several moduli."""
    matrix = [[2, 1, 3], [1, 1, 1], [4, 2, 1]]
    assert inverse_mod(matrix, 7) == [[3, 6, 6], [5, 2, 4], [6, 0, 4]]

    _check_inverse([[1, 2, 3], [0, 1, 4], [0, 0, 1]], 12)
    _check_inverse([[2, 5, 1], [1, 3, 2], [4, 0, 3]], 11)


def test_inverse_mod_generic_without_unit_pivots() -> None:
    """Exercise Bezout elimination when initial pivots are nonunits."""
    matrix = [
        [2, 3, 0, 0],
        [3, 2, 0, 0],
        [0, 0, 2, 3],
        [0, 0, 3, 2],
    ]
    assert inverse_mod(matrix, 6) == matrix


def test_inverse_mod_generated_invertible_matrices() -> None:
    """Verify generated inverses across matrix sizes and composite moduli."""
    rng = random.Random(12345)
    for n in range(2, 8):
        for m in [2, 4, 5, 6, 8, 9, 10, 12, 15]:
            for _ in range(10):
                matrix = _random_invertible_matrix(n, m, rng)
                _check_inverse(matrix, m)


def test_inverse_mod_numpy_input_and_no_mutation() -> None:
    """Accept NumPy matrices and scalar moduli without mutating the input."""
    matrix = np.array([[1, 2], [3, 4]], dtype=np.int64)
    matrix_before = matrix.copy()
    assert inverse_mod(matrix, np.int64(5)) == [[3, 1], [4, 2]]
    assert np.array_equal(matrix, matrix_before)


def test_inverse_mod_noninvertible() -> None:
    """Reject matrices whose determinants are nonunits modulo m."""
    matrices = [
        ([[1, 2], [2, 4]], 5),
        ([[1, 0, 0], [0, 2, 0], [0, 0, 1]], 6),
        ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]], 6),
    ]
    for matrix, m in matrices:
        with pytest.raises(ValueError):
            inverse_mod(matrix, m)


def test_inverse_mod_invalid_shape() -> None:
    """Reject invalid matrix shapes and moduli."""
    for matrix in [[], [[1, 2]], [[1], [2]], [[1, 0], [0, 1, 0]]]:
        with pytest.raises(ValueError):
            inverse_mod(matrix, 5)

    with pytest.raises(ValueError):
        inverse_mod([[1]], 1)
