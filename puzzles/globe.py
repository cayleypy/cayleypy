from typing import Dict

from cayleypy.cayley_graph import CayleyGraphDef
from cayleypy.permutation_utils import inverse_permutation


def help_cyclic(start_pos: int, finish_pos: int, n: int) -> list[int]:
    lst = []
    for i in range(start_pos):
        lst.append(i)
    for i in range(start_pos, finish_pos + 1):
        lst.append((i + 1) if i != finish_pos else start_pos)
    for i in range(finish_pos + 1, n):
        lst.append(i)
    return lst


def globe_gens(a: int, b: int) -> Dict[str, list[int]]:
    gens = {}
    x_count = 2 * b
    y_count = a + 1
    n = 2 * (a + 1) * b
    for r_count in range(y_count):
        gens[f"r{r_count}"] = help_cyclic(r_count * x_count, (r_count + 1) * x_count - 1, n)

    total_a = y_count - 1
    for f_count in range(x_count):
        lst = list(range(n))

        for i in range(y_count // 2):
            block1 = []
            block2 = []
            for k in range(b):
                idx1 = i * x_count + (f_count + k) % x_count
                block1.append(idx1)
                idx2 = (total_a - i) * x_count + (f_count + k) % x_count
                block2.append(idx2)
            for k in range(b):
                idx1 = block1[k]
                idx2 = block2[b - 1 - k]
                lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
        gens[f"f{f_count}"] = lst

    return gens


def globe_puzzle(a: int, b: int) -> CayleyGraphDef:
    """Cayley graph for Globe puzzle group, a + 1 cycle and 2b order 2 generators."""
    generators = []
    generator_names = []
    moves = globe_gens(a, b)
    for key, perm in moves.items():
        generators += [perm]
        generator_names += [key]
        if "r" in key:
            generators += [inverse_permutation(perm)]
            generator_names += [key + "_inv"]

    central_state = list(range(2 * b * (a + 1)))
    name = f"globe_puzzle-{a}-{b}"
    return CayleyGraphDef.create(generators, generator_names, central_state, name)
