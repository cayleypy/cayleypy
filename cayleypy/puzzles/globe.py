from typing import Dict
from cayleypy.permutation_utils import compose_permutations, apply_permutation, inverse_permutation


def help_cyclic(start_pos: int, finish_pos: int, N: int) -> list[int]:
    lst = []
    for i in range(start_pos):
        lst.append(i)
    for i in range(start_pos, finish_pos + 1):
        lst.append((i + 1) if i != finish_pos else start_pos)
    for i in range(finish_pos + 1, N):
        lst.append(i)
    return lst


def globe_gens(A: int, B: int) -> Dict[str, list[int]]:
    gens = {}
    x_count = 2 * B
    y_count = A + 1
    N = 2 * (A + 1) * B
    for r_count in range(y_count):
        gens[f"r{r_count}"] = help_cyclic(r_count * x_count, (r_count + 1) * x_count - 1, N)

    total_A = y_count - 1
    for f_count in range(x_count):
        lst = list(range(N))

        for i in range(y_count // 2):
            block1 = []
            block2 = []
            for k in range(B):
                idx1 = i * x_count + (f_count + k) % x_count
                block1.append(idx1)
                idx2 = (total_A - i) * x_count + (f_count + k) % x_count
                block2.append(idx2)
            for k in range(B):
                idx1 = block1[k]
                idx2 = block2[B - 1 - k]
                lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
        gens[f"f{f_count}"] = lst

    return gens


def full_set_of_perm_globe(A: int, B: int) -> Dict[str, list[int]]:
    original_dict = globe_gens(A, B)
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = value
        if "r" in key:
            inv_key = key + "_inv"
            new_dict[inv_key] = inverse_permutation(value)
    return new_dict
