import random

from cayleypy.permutation_utils import permutation_between, apply_permutation


def test_permutation_between():
    a = [random.randint(0, 20) for _ in range(50)]
    b = list(a)
    random.shuffle(b)
    p = permutation_between(a, b)
    assert apply_permutation(p, a) == b
