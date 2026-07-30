import math

import numpy as np
import pytest
import torch

from .permutation_utils import apply_permutation
from .string_encoder import StringEncoder


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100), (4, 16), (1, 64)])
def test_encode_decode(code_width, n):
    num_states = 5
    s = torch.randint(0, 2**code_width, (num_states, n))
    enc = StringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    assert s_encoded.shape == (num_states, int(math.ceil(code_width * n / 64)))
    assert torch.equal(s, enc.decode(s_encoded))


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100), (4, 16), (1, 64)])
def test_permutation(code_width: int, n: int):
    num_states = 5
    s = torch.randint(0, 2**code_width, (num_states, n), dtype=torch.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    expected = torch.tensor([apply_permutation(perm, row) for row in s.numpy()], dtype=torch.int64)
    enc = StringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    result = torch.zeros_like(s_encoded)
    perm_func = enc.implement_permutation(perm)
    perm_func(s_encoded, result)
    ans = enc.decode(result)
    assert torch.equal(ans, expected)


def test_permutation_cross_codeword_boundary():
    """B5 disproven: bit 63 of a codeword can never be shifted left (shift>0).

    When a mask includes bit 63 (sign bit), shift is always ≤ 0 because
    start_bit % 64 = 63 is the highest position — any left shift would exceed
    the codeword. The shift < 0 branch already handles mask < 0 correctly.
    This test verifies correctness for n*w > 64 (multi-codeword states) with
    random permutations, confirming the bug does not exist.
    """
    num_states = 10
    for code_width, n in [(1, 65), (2, 33), (3, 22), (4, 17), (1, 70), (5, 20)]:
        s = torch.randint(0, 2**code_width, (num_states, n), dtype=torch.int64)
        perm = [int(x) for x in np.random.permutation(n)]
        expected = torch.tensor([apply_permutation(perm, row) for row in s.numpy()], dtype=torch.int64)
        enc = StringEncoder(code_width=code_width, n=n)
        s_encoded = enc.encode(s)
        result = torch.zeros_like(s_encoded)
        perm_func = enc.implement_permutation(perm)
        perm_func(s_encoded, result)
        ans = enc.decode(result)
        assert torch.equal(ans, expected), f"Failed for w={code_width}, n={n}"


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (4, 16), (1, 64)])
def test_permutation_1d(code_width: int, n: int):
    num_states = 5
    s = torch.randint(0, 2**code_width, (num_states, n), dtype=torch.int64)
    perm = [int(x) for x in np.random.permutation(n)]
    expected = torch.tensor([apply_permutation(perm, row) for row in s.numpy()], dtype=torch.int64)
    enc = StringEncoder(code_width=code_width, n=n)
    perm_func = enc.implement_permutation_1d(perm)
    ans = enc.decode(perm_func(enc.encode(s)))  # type: ignore
    assert torch.equal(ans, expected)


# =============================================================================
# C17: StringEncoder edge cases
# =============================================================================


def test_implement_permutation_1d_encoded_length_gt_one_raises():
    """C17: implement_permutation_1d raises AssertionError when encoded_length > 1."""
    enc = StringEncoder(code_width=4, n=30)  # 4*30=120 bits -> 2 codewords
    assert enc.encoded_length > 1
    perm = list(range(30))
    with pytest.raises(AssertionError):
        enc.implement_permutation_1d(perm)


def test_encode_negative_values_raises():
    """C17: encode with negative values raises AssertionError."""
    enc = StringEncoder(code_width=1, n=5)
    s = torch.tensor([[0, -1, 0, 1, 0]], dtype=torch.int64)
    with pytest.raises(AssertionError, match="negative"):
        enc.encode(s)
