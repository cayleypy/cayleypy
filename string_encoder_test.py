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
