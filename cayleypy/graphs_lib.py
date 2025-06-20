"""Library of pre-defined graphs."""

from cayleypy.cayley_graph import CayleyGraphDef
from cayleypy.hungarian_rings import hungarian_rings_generators
from cayleypy.permutation_utils import (
    compose_permutations,
    transposition,
    permutation_from_cycles as pfc,
    inverse_permutation,
)

CUBE222_MOVES = {
    "f0": pfc(24, [[2, 19, 21, 8], [3, 17, 20, 10], [4, 6, 7, 5]]),
    "r1": pfc(24, [[1, 5, 21, 14], [3, 7, 23, 12], [8, 10, 11, 9]]),
    "d0": pfc(24, [[6, 18, 14, 10], [7, 19, 15, 11], [20, 22, 23, 21]]),
}

CUBE333_MOVES = {
    "U": pfc(54, [[0, 6, 8, 2], [1, 3, 7, 5], [20, 47, 29, 38], [23, 50, 32, 41], [26, 53, 35, 44]]),
    "D": pfc(54, [[9, 15, 17, 11], [10, 12, 16, 14], [18, 36, 27, 45], [21, 39, 30, 48], [24, 42, 33, 51]]),
    "L": pfc(54, [[0, 44, 9, 45], [1, 43, 10, 46], [2, 42, 11, 47], [18, 24, 26, 20], [19, 21, 25, 23]]),
    "R": pfc(54, [[6, 51, 15, 38], [7, 52, 16, 37], [8, 53, 17, 36], [27, 33, 35, 29], [28, 30, 34, 32]]),
    "B": pfc(54, [[2, 35, 15, 18], [5, 34, 12, 19], [8, 33, 9, 20], [36, 42, 44, 38], [37, 39, 43, 41]]),
    "F": pfc(54, [[0, 24, 17, 29], [3, 25, 14, 28], [6, 26, 11, 27], [45, 51, 53, 47], [46, 48, 52, 50]]),
}

MINI_PARAMORPHIX_ALLOWED_MOVES = {
    "M_DF": [0, 1, 2, 3, 4, 5, 11, 9, 10, 7, 8, 6, 15, 16, 17, 12, 13, 14, 18, 19, 20, 21, 22, 23],
    "M_RL": [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 18, 19, 20],
    "M_DFv": [3, 4, 5, 0, 1, 2, 11, 9, 10, 7, 8, 6, 15, 16, 17, 12, 13, 14, 21, 22, 23, 18, 19, 20],
    "M_LF": [8, 6, 7, 3, 4, 5, 1, 2, 0, 9, 10, 11, 18, 19, 20, 15, 16, 17, 12, 13, 14, 21, 22, 23],
    "M_RD": [0, 1, 2, 10, 11, 9, 6, 7, 8, 5, 3, 4, 12, 13, 14, 21, 22, 23, 18, 19, 20, 15, 16, 17],
    "M_LFv": [8, 6, 7, 10, 11, 9, 1, 2, 0, 5, 3, 4, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17],
    "M_DL": [0, 1, 2, 8, 6, 7, 4, 5, 3, 9, 10, 11, 12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23],
    "M_FR": [10, 11, 9, 3, 4, 5, 6, 7, 8, 2, 0, 1, 21, 22, 23, 15, 16, 17, 18, 19, 20, 12, 13, 14],
    "M_DLv": [10, 11, 9, 8, 6, 7, 4, 5, 3, 2, 0, 1, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14],
    "M_Fv": [7, 8, 6, 5, 3, 4, 10, 11, 9, 1, 2, 0, 13, 14, 12, 22, 23, 21, 16, 17, 15, 19, 20, 18],
    "M_Fvi": [11, 9, 10, 4, 5, 3, 2, 0, 1, 8, 6, 7, 14, 12, 13, 20, 18, 19, 23, 21, 22, 17, 15, 16],
    "M_Dv": [2, 0, 1, 9, 10, 11, 3, 4, 5, 6, 7, 8, 19, 20, 18, 16, 17, 15, 22, 23, 21, 13, 14, 12],
    "M_Dvi": [1, 2, 0, 6, 7, 8, 9, 10, 11, 3, 4, 5, 23, 21, 22, 17, 15, 16, 14, 12, 13, 20, 18, 19],
    "M_Lv": [5, 3, 4, 7, 8, 6, 0, 1, 2, 11, 9, 10, 22, 23, 21, 13, 14, 12, 19, 20, 18, 16, 17, 15],
    "M_Lvi": [6, 7, 8, 1, 2, 0, 5, 3, 4, 10, 11, 9, 17, 15, 16, 23, 21, 22, 20, 18, 19, 14, 12, 13],
    "M_Rv": [9, 10, 11, 2, 0, 1, 8, 6, 7, 4, 5, 3, 16, 17, 15, 19, 20, 18, 13, 14, 12, 22, 23, 21],
    "M_Rvi": [4, 5, 3, 11, 9, 10, 7, 8, 6, 0, 1, 2, 20, 18, 19, 14, 12, 13, 17, 15, 16, 23, 21, 22],
}


def _create_coxeter_generators(n: int) -> list[list[int]]:
    return [transposition(n, k, k + 1) for k in range(n - 1)]


def _create_cyclic_coxeter_generators(n: int) -> list[list[int]]:
    return _create_coxeter_generators(n) + [transposition(n, 0, n - 1)]


def prepare_graph(name: str, n: int = 0, **kwargs) -> CayleyGraphDef:
    """Returns pre-defined Cayley or Schreier coset graph.

    Supported graphs:
      * "all_transpositions" - Cayley graph for S_n (n>=2), generated by all n(n-1)/2 transpositions.
      * "pancake" - Cayley graph for S_n (n>=2), generated by reverses of all prefixes. It has n-1 generators denoted
          R1,R2..R(n-1), where Ri is reverse of elements 0,1..i. See https://en.wikipedia.org/wiki/Pancake_graph.
      * "burnt_pancake" - Cayley graph generated by reverses of all signed prefixes. Actually is a graph for
         S_2n (n>=1) representing a graph for n pancakes, where i-th element represents bottom side of i-th pancake,
         and (n+i)-th element represents top side of i-th pancake. The graph has n generators denoted R1,R2..R(n),
         where Ri is reverse of elements 0,1..i,n,n+1..n+i.
      * "full_reversals" - Cayley graph for S_n (n>=2), generated by reverses of all possible substrings.
          It has n(n-1)/2 generators.
      * "lrx" - Cayley graph for S_n (n>=3), generated by: shift left, shift right, swap two elements 0 and k.
          Has parameter k (1<=k<n). It specifies that X is transposition of elements 0 and k.
          By default, k=1, which means X is transposition of first 2 elements.
      * "top_spin" - Cayley graph for S_n (n>=k), generated by: shift left, shift right, reverse first k elements.
          Has parameter k. By default, k=4.
      * "cube_2/2/2_6gensQTM" - Schreier coset graph for 2x2x2 Rubik's cube with fixed back left upper corner and only
          quarter-turns allowed. There are 6 generators (front, right, down face - clockwise and counterclockwise).
      * "cube_2/2/2_9gensHTM" - same as above, but allowing half-turns (it has 9 generators).
      * "cube_3/3/3_12gensQTM" - Schreier coset graph for 3x3x3 Rubik's cube with fixed central pieces and only
          quarter-turns allowed. There are 12 generators (clockwise and counterclockwise rotation for each face).
      * "cube_3/3/3_18gensHTM" - same as above, but allowing half-turns (it has 18 generators).
      * "coxeter" - Cayley graph for S_n (n>=2), generated by adjacent transpositions (Coxeter generators).
          It has n-1 generators: (0,1), (1,2), ..., (n-2,n-1).
      * "cyclic_coxeter" - Cayley graph for S_n (n>=2), generated by adjacent transpositions plus cyclic transposition.
          It has n generators: (0,1), (1,2), ..., (n-2,n-1), (0,n-1).
      * "mini_paramorphix" – Cayley graph for a subgroup of S_24, acting on 24 titles. It is generated by 17 moves
          inspired by a simplified version of the Paramorphix puzzle. Moves are based on overlapping 2- and 3-cycles
          and result in a symmetric, undirected graph. (order of the graph 24, degree 17, order of the group 288)
      * "hungarian_rings" - Cayley graph for S_n (n>=4), generated by rotating two rings in both directions.
          For each ring structure and their intersection it has four generators.

    :param name: name of pre-defined graph.
    :param n: parameter (if applicable).
    :return: requested graph as `CayleyGraph`.
    """
    if name == "all_transpositions":
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                generators.append(transposition(n, i, j))
                generator_names.append(f"({i},{j})")
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    elif name == "pancake":
        assert n >= 2
        generators = []
        generator_names = []
        for prefix_len in range(2, n + 1):
            perm = list(range(prefix_len - 1, -1, -1)) + list(range(prefix_len, n))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len - 1))
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    elif name == "burnt_pancake":
        assert n >= 1
        generators = []
        generator_names = []
        for prefix_len in range(0, n):
            perm = []
            perm += list(range(n + prefix_len, n - 1, -1))
            perm += list(range(prefix_len + 1, n, 1))
            perm += list(range(prefix_len, -1, -1))
            perm += list(range(n + prefix_len + 1, 2 * n, 1))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len + 1))
        return CayleyGraphDef.create(generators, central_state=list(range(2 * n)), generator_names=generator_names)
    elif name == "full_reversals":
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                perm = list(range(i)) + list(range(j, i - 1, -1)) + list(range(j + 1, n))
                generators.append(perm)
                generator_names.append(f"R[{i}..{j}]")
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    elif name == "lrx":
        assert n >= 3
        k = kwargs.get("k", 1)
        generators = [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), transposition(n, 0, k)]
        generator_names = ["L", "R", "X"]
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    elif name == "top_spin":
        k = kwargs.get("k", 4)
        assert n >= k
        generators = [
            list(range(1, n)) + [0],
            [n - 1] + list(range(0, n - 1)),
            list(range(k - 1, -1, -1)) + list(range(k, n)),
        ]
        return CayleyGraphDef.create(generators, central_state=list(range(n)))
    elif name == "cube_2/2/2_6gensQTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE222_MOVES.items():
            generators += [perm, inverse_permutation(perm)]
            generator_names += [move_id, move_id + "'"]
        central_state = [color for color in range(6) for _ in range(4)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cube_2/2/2_9gensHTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE222_MOVES.items():
            generators += [perm, inverse_permutation(perm), compose_permutations(perm, perm)]
            generator_names += [move_id, move_id + "'", move_id + "^2"]
        central_state = [color for color in range(6) for _ in range(4)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cube_3/3/3_12gensQTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE333_MOVES.items():
            generators += [perm, inverse_permutation(perm)]
            generator_names += [move_id, move_id + "'"]
        central_state = [color for color in range(6) for _ in range(9)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cube_3/3/3_18gensHTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE333_MOVES.items():
            generators += [perm, inverse_permutation(perm), compose_permutations(perm, perm)]
            generator_names += [move_id, move_id + "'", move_id + "^2"]
        central_state = [color for color in range(6) for _ in range(9)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "coxeter":
        assert n >= 2
        generators = _create_coxeter_generators(n)
        generator_names = [f"({i},{i + 1})" for i in range(n - 1)]
        central_state = list(range(n))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cyclic_coxeter":
        assert n >= 2
        generators = _create_cyclic_coxeter_generators(n)
        generator_names = [f"({i},{i + 1})" for i in range(n - 1)] + [f"(0,{n - 1})"]
        central_state = list(range(n))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "mini_paramorphix":
        generator_names = list(MINI_PARAMORPHIX_ALLOWED_MOVES.keys())
        generators = [MINI_PARAMORPHIX_ALLOWED_MOVES[k] for k in generator_names]
        central_state = list(range(len(generators[0])))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "hungarian_rings":
        assert n % 2 == 0
        ring_size = (n + 2) // 2
        assert ring_size >= 4
        generators, generator_names = hungarian_rings_generators(ring_size=ring_size)
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    else:
        raise ValueError(f"Unknown generator set: {name}")
