"""Helper functions for matrices."""

from math import gcd


def _bezout(a, b):
    g = gcd(a, b)
    if b == g:
        return g, 0, 1
    s = pow(a // g, -1, b // g)
    t = (g - s * a) // b
    return g, s, t


def inverse_integer(A):
    A = [[int(x) for x in row] for row in A]
    n = len(A)

    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a nonempty square matrix")

    aug = [
        row + [int(i == j) for j in range(n)]
        for i, row in enumerate(A)
    ]

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

                aug[k] = [
                    s * x + t * y
                    for x, y in zip(row_k, row_i)
                ]
                aug[i] = [
                    -(b // g) * x + (a // g) * y
                    for x, y in zip(row_k, row_i)
                ]

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
                aug[i] = [
                    x - q * y
                    for x, y in zip(aug[i], aug[k])
                ]

    return [row[n:] for row in aug]


def _clean(A, m, size=None):
    """Reduce entries modulo m and perform the useful shape checks."""
    if m < 2:
        raise ValueError("modulus m must be at least 2")
    A = [[int(x) % m for x in row] for row in A]
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a nonempty square matrix")
    if size is not None and n != size:
        raise ValueError(f"expected a {size}x{size} matrix, got {n}x{n}")
    return A


def _inverse_2x2(A, m):
    (a, b), (c, d) = A
    s = pow((a * d - b * c) % m, -1, m)
    return [
        [s * d % m, -s * b % m],
        [-s * c % m, s * a % m],
    ]


def _inverse_3x3(A, m):
    (a, b, c), (d, e, f), (g, h, i) = A

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


def _inverse_generic(A, m):
    """Factorization-free Bezout--Gauss--Jordan elimination."""
    n = len(A)
    M = [
        row[:] + [int(i == j) for j in range(n)]
        for i, row in enumerate(A)
    ]

    for k in range(n):
        # Use an existing unit pivot when possible.
        pivot_row = next(
            (i for i in range(k, n) if gcd(M[i][k], m) == 1),
            None,
        )
        if pivot_row is not None:
            M[k], M[pivot_row] = M[pivot_row], M[k]
        else:
            # Over composite moduli, several nonunits may combine into a unit.
            for i in range(k + 1, n):
                if gcd(M[k][k], m) == 1:
                    break
                b = M[i][k]
                if b == 0:
                    continue

                a = M[k][k]
                g, s, t = _bezout(a, b)
                R, S = M[k][:], M[i][:]
                M[k] = [(s * x + t * y) % m for x, y in zip(R, S)]
                M[i] = [
                    (-(b // g) * x + (a // g) * y) % m
                    for x, y in zip(R, S)
                ]

        pivot = M[k][k]
        if gcd(pivot, m) != 1:
            raise ValueError(f"matrix is not invertible modulo {m}")

        s = pow(pivot, -1, m)
        M[k] = [s * x % m for x in M[k]]

        for i in range(n):
            if i != k and M[i][k]:
                s = M[i][k]
                M[i] = [(x - s * y) % m for x, y in zip(M[i], M[k])]

    return [row[n:] for row in M]


def inverse_mod(A, m, method="auto", orientation="auto"):
    """Invert A modulo m, selecting an appropriate implementation."""
    A = _clean(A, m)
    n = len(A)

    if method == "auto":
        if n == 2:
            return _inverse_2x2(A, m)
        if n == 3:
            return _inverse_3x3(A, m)
        return _inverse_generic(A, m)

    if method == "2x2":
        if n != 2:
            raise ValueError("method='2x2' requires a 2x2 matrix")
        return _inverse_2x2(A, m)
    if method == "3x3":
        if n != 3:
            raise ValueError("method='3x3' requires a 3x3 matrix")
        return _inverse_3x3(A, m)
    if method == "generic":
        return _inverse_generic(A, m)

    raise ValueError(
        "method must be 'auto', '2x2', '3x3', or 'generic'"
    )
