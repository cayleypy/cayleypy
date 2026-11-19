"""Helper functions for matrices."""

from math import gcd
from typing import Iterable, Optional, SupportsInt


def _bezout(a: int, b: int) -> tuple[int, int, int]:
    """Return the gcd and coefficients of a Bezout identity."""
    g = gcd(a, b)
    if b == g:
        return g, 0, 1
    s = pow(a // g, -1, b // g)
    t = (g - s * a) // b
    return g, s, t


def _inverse_integer_2x2(matrix: list[list[int]]) -> list[list[int]]:
    """Invert a unimodular 2x2 matrix using its adjugate."""
    (a, b), (c, d) = matrix
    determinant = a * d - b * c

    if abs(determinant) != 1:
        raise ValueError("matrix is not invertible over Z")

    return [
        [determinant * d, -determinant * b],
        [-determinant * c, determinant * a],
    ]


def _inverse_integer_3x3(matrix: list[list[int]]) -> list[list[int]]:
    """Invert a unimodular 3x3 matrix using its adjugate."""
    (a, b, c), (d, e, f), (g, h, i) = matrix

    x = e * i - f * h
    y = f * g - d * i
    z = d * h - e * g
    determinant = a * x + b * y + c * z

    if abs(determinant) != 1:
        raise ValueError("matrix is not invertible over Z")

    adjugate = [
        [x, c * h - b * i, b * f - c * e],
        [y, a * i - c * g, c * d - a * f],
        [z, b * g - a * h, a * e - b * d],
    ]
    return [[determinant * value for value in row] for row in adjugate]


def _inverse_integer_generic(matrix: list[list[int]]) -> list[list[int]]:
    """Invert a unimodular matrix using integer row operations."""
    n = len(matrix)

    aug = [row + [int(i == j) for j in range(n)] for i, row in enumerate(matrix)]

    for k in range(n):
        # Prefer an existing unit pivot.
        pivot_row = next(
            (i for i in range(k, n) if abs(aug[i][k]) == 1),
            None,
        )

        if pivot_row is not None:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
        else:
            # Unimodular row operations reduce the column gcd.
            for i in range(k + 1, n):
                a, b = aug[k][k], aug[i][k]
                if b == 0:
                    continue

                g, s, t = _bezout(abs(a), abs(b))
                if a < 0:
                    s = -s
                if b < 0:
                    t = -t

                row_k = aug[k]
                row_i = aug[i]

                aug[k] = [s * x + t * y for x, y in zip(row_k, row_i)]
                aug[i] = [-(b // g) * x + (a // g) * y for x, y in zip(row_k, row_i)]

                if abs(aug[k][k]) == 1:
                    break

        if abs(aug[k][k]) != 1:
            raise ValueError("Matrix is not invertible over Z")

        if aug[k][k] == -1:
            aug[k] = [-x for x in aug[k]]

        for i in range(n):
            if i == k:
                continue

            q = aug[i][k]
            if q:
                aug[i] = [x - q * y for x, y in zip(aug[i], aug[k])]

    return [row[n:] for row in aug]


def inverse_integer(matrix: Iterable[Iterable[SupportsInt]]) -> list[list[int]]:
    """Return the exact integer inverse of a unimodular square matrix."""
    matrix = [[int(x) for x in row] for row in matrix]
    n = len(matrix)

    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("A must be a nonempty square matrix")

    if n == 2:
        return _inverse_integer_2x2(matrix)
    if n == 3:
        return _inverse_integer_3x3(matrix)

    return _inverse_integer_generic(matrix)


def _clean(matrix: Iterable[Iterable[SupportsInt]], m: int, size: Optional[int] = None) -> list[list[int]]:
    """Reduce entries modulo m and perform the useful shape checks."""
    if m < 2:
        raise ValueError("modulus m must be at least 2")
    matrix = [[int(x) % m for x in row] for row in matrix]
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("A must be a nonempty square matrix")
    if size is not None and n != size:
        raise ValueError(f"expected a {size}x{size} matrix, got {n}x{n}")
    return matrix


def _inverse_2x2(matrix: list[list[int]], m: int) -> list[list[int]]:
    """Invert a 2x2 matrix modulo m using its adjugate."""
    (a, b), (c, d) = matrix
    s = pow((a * d - b * c) % m, -1, m)
    return [
        [s * d % m, -s * b % m],
        [-s * c % m, s * a % m],
    ]


def _inverse_3x3(matrix: list[list[int]], m: int) -> list[list[int]]:
    """Invert a 3x3 matrix modulo m using its adjugate."""
    (a, b, c), (d, e, f), (g, h, i) = matrix

    x = e * i - f * h
    y = f * g - d * i
    z = d * h - e * g
    s = pow((a * x + b * y + c * z) % m, -1, m)

    adjugate = [
        [x, c * h - b * i, b * f - c * e],
        [y, a * i - c * g, c * d - a * f],
        [z, b * g - a * h, a * e - b * d],
    ]
    return [[s * x % m for x in row] for row in adjugate]


def _inverse_generic(matrix: list[list[int]], m: int) -> list[list[int]]:
    """Factorization-free Bezout--Gauss--Jordan elimination."""
    n = len(matrix)
    aug = [row[:] + [int(i == j) for j in range(n)] for i, row in enumerate(matrix)]

    for k in range(n):
        # Use an existing unit pivot when possible.
        pivot_row = next(
            (i for i in range(k, n) if gcd(aug[i][k], m) == 1),
            None,
        )
        if pivot_row is not None:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
        else:
            # Over composite moduli, several nonunits may combine into a unit.
            for i in range(k + 1, n):
                if gcd(aug[k][k], m) == 1:
                    break
                b = aug[i][k]
                if b == 0:
                    continue

                a = aug[k][k]
                g, s, t = _bezout(a, b)
                row_k, row_i = aug[k][:], aug[i][:]
                aug[k] = [(s * x + t * y) % m for x, y in zip(row_k, row_i)]
                aug[i] = [(-(b // g) * x + (a // g) * y) % m for x, y in zip(row_k, row_i)]

        pivot = aug[k][k]
        if gcd(pivot, m) != 1:
            raise ValueError(f"matrix is not invertible modulo {m}")

        s = pow(pivot, -1, m)
        aug[k] = [s * x % m for x in aug[k]]

        for i in range(n):
            if i != k and aug[i][k]:
                s = aug[i][k]
                aug[i] = [(x - s * y) % m for x, y in zip(aug[i], aug[k])]

    return [row[n:] for row in aug]


def inverse_mod(
    matrix: Iterable[Iterable[SupportsInt]],
    m: int,
    method: str = "auto",
    orientation: str = "auto",  # pylint: disable=unused-argument
) -> list[list[int]]:
    """Invert a matrix modulo m; orientation is retained for compatibility only."""
    matrix = _clean(matrix, m)
    n = len(matrix)

    if method == "auto":
        if n == 2:
            return _inverse_2x2(matrix, m)
        if n == 3:
            return _inverse_3x3(matrix, m)
        return _inverse_generic(matrix, m)

    if method == "2x2":
        if n != 2:
            raise ValueError("method='2x2' requires a 2x2 matrix")
        return _inverse_2x2(matrix, m)
    if method == "3x3":
        if n != 3:
            raise ValueError("method='3x3' requires a 3x3 matrix")
        return _inverse_3x3(matrix, m)
    if method == "generic":
        return _inverse_generic(matrix, m)

    raise ValueError("method must be 'auto', '2x2', '3x3', or 'generic'")
