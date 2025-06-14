"""Library of pre-defined graphs."""
from cayleypy import CayleyGraph
from cayleypy.permutation_utils import compose_permutations, apply_permutation

CUBE222_ALLOWED_MOVES = {
    'f0': [0, 1, 19, 17, 6, 4, 7, 5, 2, 9, 3, 11, 12, 13, 14, 15, 16, 20, 18, 21, 10, 8, 22, 23],
    '-f0': [0, 1, 8, 10, 5, 7, 4, 6, 21, 9, 20, 11, 12, 13, 14, 15, 16, 3, 18, 2, 17, 19, 22, 23],
    'r1': [0, 5, 2, 7, 4, 21, 6, 23, 10, 8, 11, 9, 3, 13, 1, 15, 16, 17, 18, 19, 20, 14, 22, 12],
    '-r1': [0, 14, 2, 12, 4, 1, 6, 3, 9, 11, 8, 10, 23, 13, 21, 15, 16, 17, 18, 19, 20, 5, 22, 7],
    'd0': [0, 1, 2, 3, 4, 5, 18, 19, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 22, 20, 23, 21],
    '-d0': [0, 1, 2, 3, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 6, 7, 21, 23, 20, 22]
}

CUBE333_ALLOWED_MOVES = {
    'U': [6, 3, 0, 7, 4, 1, 8, 5, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 47, 21, 22, 50, 24, 25, 53, 27, 28, 38,
          30, 31, 41, 33, 34, 44, 36, 37, 20, 39, 40, 23, 42, 43, 26, 45, 46, 29, 48, 49, 32, 51, 52, 35],
    'D': [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 36, 19, 20, 39, 22, 23, 42, 25, 26, 45, 28, 29,
          48, 31, 32, 51, 34, 35, 27, 37, 38, 30, 40, 41, 33, 43, 44, 18, 46, 47, 21, 49, 50, 24, 52, 53],
    'L': [44, 43, 42, 3, 4, 5, 6, 7, 8, 45, 46, 47, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 11, 10, 9, 0, 1, 2, 48, 49, 50, 51, 52, 53],
    'R': [0, 1, 2, 3, 4, 5, 51, 52, 53, 9, 10, 11, 12, 13, 14, 38, 37, 36, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33, 30,
          27, 34, 31, 28, 35, 32, 29, 8, 7, 6, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 15, 16, 17],
    'B': [0, 1, 35, 3, 4, 34, 6, 7, 33, 20, 10, 11, 19, 13, 14, 18, 16, 17, 2, 5, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          30, 31, 32, 9, 12, 15, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'F': [24, 1, 2, 25, 4, 5, 26, 7, 8, 9, 10, 27, 12, 13, 28, 15, 16, 29, 18, 19, 20, 21, 22, 23, 17, 14, 11, 6, 3, 0,
          30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47],
    "U'": [2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 38, 21, 22, 41, 24, 25, 44, 27, 28, 47,
           30, 31, 50, 33, 34, 53, 36, 37, 29, 39, 40, 32, 42, 43, 35, 45, 46, 20, 48, 49, 23, 51, 52, 26],
    "D'": [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 45, 19, 20, 48, 22, 23, 51, 25, 26, 36, 28, 29,
           39, 31, 32, 42, 34, 35, 18, 37, 38, 21, 40, 41, 24, 43, 44, 27, 46, 47, 30, 49, 50, 33, 52, 53],
    "L'": [45, 46, 47, 3, 4, 5, 6, 7, 8, 44, 43, 42, 12, 13, 14, 15, 16, 17, 20, 23, 26, 19, 22, 25, 18, 21, 24, 27, 28,
           29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 2, 1, 0, 9, 10, 11, 48, 49, 50, 51, 52, 53],
    "R'": [0, 1, 2, 3, 4, 5, 38, 37, 36, 9, 10, 11, 12, 13, 14, 51, 52, 53, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 32,
           35, 28, 31, 34, 27, 30, 33, 17, 16, 15, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 6, 7, 8],
    "B'": [0, 1, 18, 3, 4, 19, 6, 7, 20, 33, 10, 11, 34, 13, 14, 35, 16, 17, 15, 12, 9, 21, 22, 23, 24, 25, 26, 27, 28,
           29, 30, 31, 32, 8, 5, 2, 38, 41, 44, 37, 40, 43, 36, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "F'": [29, 1, 2, 28, 4, 5, 27, 7, 8, 9, 10, 26, 12, 13, 25, 15, 16, 24, 18, 19, 20, 21, 22, 23, 0, 3, 6, 11, 14, 17,
           30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51],
}

MINI_PARAMORPHIX_ALLOWED_MOVES = {
    "M_DF":  [0, 1, 2, 3, 4, 5, 11, 9, 10, 7, 8, 6, 15, 16, 17, 12, 13, 14, 18, 19, 20, 21, 22, 23],
    "M_RL":  [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 18, 19, 20],
    "M_DFv": [3, 4, 5, 0, 1, 2, 11, 9, 10, 7, 8, 6, 15, 16, 17, 12, 13, 14, 21, 22, 23, 18, 19, 20],
    "M_LF":  [8, 6, 7, 3, 4, 5, 1, 2, 0, 9, 10, 11, 18, 19, 20, 15, 16, 17, 12, 13, 14, 21, 22, 23],
    "M_RD":  [0, 1, 2, 10, 11, 9, 6, 7, 8, 5, 3, 4, 12, 13, 14, 21, 22, 23, 18, 19, 20, 15, 16, 17],
    "M_LFv": [8, 6, 7, 10, 11, 9, 1, 2, 0, 5, 3, 4, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17],
    "M_DL":  [0, 1, 2, 8, 6, 7, 4, 5, 3, 9, 10, 11, 12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23],
    "M_FR":  [10, 11, 9, 3, 4, 5, 6, 7, 8, 2, 0, 1, 21, 22, 23, 15, 16, 17, 18, 19, 20, 12, 13, 14],
    "M_DLv": [10, 11, 9, 8, 6, 7, 4, 5, 3, 2, 0, 1, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14],
    "M_Fv":  [7, 8, 6, 5, 3, 4, 10, 11, 9, 1, 2, 0, 13, 14, 12, 22, 23, 21, 16, 17, 15, 19, 20, 18],
    "M_Fvi": [11, 9, 10, 4, 5, 3, 2, 0, 1, 8, 6, 7, 14, 12, 13, 20, 18, 19, 23, 21, 22, 17, 15, 16],
    "M_Dv":  [2, 0, 1, 9, 10, 11, 3, 4, 5, 6, 7, 8, 19, 20, 18, 16, 17, 15, 22, 23, 21, 13, 14, 12],
    "M_Dvi": [1, 2, 0, 6, 7, 8, 9, 10, 11, 3, 4, 5, 23, 21, 22, 17, 15, 16, 14, 12, 13, 20, 18, 19],
    "M_Lv":  [5, 3, 4, 7, 8, 6, 0, 1, 2, 11, 9, 10, 22, 23, 21, 13, 14, 12, 19, 20, 18, 16, 17, 15],
    "M_Lvi": [6, 7, 8, 1, 2, 0, 5, 3, 4, 10, 11, 9, 17, 15, 16, 23, 21, 22, 20, 18, 19, 14, 12, 13],
    "M_Rv":  [9, 10, 11, 2, 0, 1, 8, 6, 7, 4, 5, 3, 16, 17, 15, 19, 20, 18, 13, 14, 12, 22, 23, 21],
    "M_Rvi": [4, 5, 3, 11, 9, 10, 7, 8, 6, 0, 1, 2, 20, 18, 19, 14, 12, 13, 17, 15, 16, 23, 21, 22]
}


def _create_coxeter_generators(n: int) -> list[list[int]]:
    gens = []
    for k in range(n - 1):
        perm = list(range(n))
        perm[k], perm[k + 1] = perm[k + 1], perm[k]
        gens.append(perm)
    return gens


def _create_cyclic_coxeter_generators(n: int) -> list[list[int]]:
    gens = []
    for k in range(n - 1):
        perm = list(range(n))
        perm[k], perm[k + 1] = perm[k + 1], perm[k]
        gens.append(perm)
    perm = list(range(n))
    perm[0], perm[n-1] = perm[n-1], perm[0]
    gens.append(perm)
    return gens


def prepare_graph(name, n=0) -> CayleyGraph:
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
      * "lrx" - Cayley graph for S_n (n>=3), generated by: shift left, shift right, swap first two elements.
      * "top_spin" - Cayley graph for S_n (n>=4), generated by: shift left, shift right, reverse first four elements.
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
                perm = list(range(n))
                perm[i], perm[j] = j, i
                generators.append(perm)
                generator_names.append(f"({i},{j})")
        return CayleyGraph(generators, dest=list(range(n)), generator_names=generator_names)
    elif name == "pancake":
        assert n >= 2
        generators = []
        generator_names = []
        for prefix_len in range(2, n + 1):
            perm = list(range(prefix_len - 1, -1, -1)) + list(range(prefix_len, n))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len - 1))
        return CayleyGraph(generators, dest=list(range(n)), generator_names=generator_names)
    elif name == "burnt_pancake":
        assert n >= 1
        generators = []
        generator_names = []
        for prefix_len in range(0, n):
            perm = []
            perm += list(range(n+prefix_len, n-1, -1))
            perm += list(range(prefix_len+1, n, 1))
            perm += list(range(prefix_len, -1, -1))
            perm += list(range(n+prefix_len+1, 2*n, 1))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len+1))
        return CayleyGraph(generators, dest=list(range(2*n)), generator_names=generator_names)
    elif name == "full_reversals":
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                perm = list(range(i)) + list(range(j, i - 1, -1)) + list(range(j + 1, n))
                generators.append(perm)
                generator_names.append(f"R[{i}..{j}]")
        return CayleyGraph(generators, dest=list(range(n)), generator_names=generator_names)
    elif name == "lrx":
        assert n >= 3
        generators = [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [1, 0] + list(range(2, n))]
        generator_names = ["L", "R", "X"]
        return CayleyGraph(generators, dest=list(range(n)), generator_names=generator_names)
    elif name == "top_spin":
        assert n >= 4
        generators = [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [3, 2, 1, 0] + list(range(4, n))]
        return CayleyGraph(generators, dest=list(range(n)))
    elif name == "cube_2/2/2_6gensQTM":
        generator_names = list(CUBE222_ALLOWED_MOVES.keys())
        generators = [CUBE222_ALLOWED_MOVES[k] for k in generator_names]
        initial_state = [color for color in range(6) for _ in range(4)]
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "cube_2/2/2_9gensHTM":
        generator_names = list(CUBE222_ALLOWED_MOVES.keys())
        generators = [CUBE222_ALLOWED_MOVES[k] for k in generator_names]
        for move_id in ['f0', 'r1', 'd0']:
            generators.append(compose_permutations(CUBE222_ALLOWED_MOVES[move_id], CUBE222_ALLOWED_MOVES[move_id]))
            generator_names.append(move_id + "^2")
        initial_state = [color for color in range(6) for _ in range(4)]
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "cube_3/3/3_12gensQTM":
        generator_names = list(CUBE333_ALLOWED_MOVES.keys())
        generators = [CUBE333_ALLOWED_MOVES[k] for k in generator_names]
        initial_state = [color for color in range(6) for _ in range(9)]
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "cube_3/3/3_18gensHTM":
        generator_names = list(CUBE333_ALLOWED_MOVES.keys())
        generators = [CUBE333_ALLOWED_MOVES[k] for k in generator_names]
        for move_id in ['U', 'D', 'L', 'R', 'B', 'F']:
            generators.append(compose_permutations(CUBE333_ALLOWED_MOVES[move_id], CUBE333_ALLOWED_MOVES[move_id]))
            generator_names.append(move_id + "^2")
        initial_state = [color for color in range(6) for _ in range(9)]
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "coxeter":
        assert n >= 2
        generators = _create_coxeter_generators(n)
        generator_names = [f"({i},{i+1})" for i in range(n-1)]
        initial_state = list(range(n))
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "cyclic_coxeter":
        assert n >= 2
        generators = _create_cyclic_coxeter_generators(n)
        generator_names = [f"({i},{i+1})" for i in range(n-1)] + [f"(0,{n-1})"]
        initial_state = list(range(n))
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "mini_paramorphix":
        generator_names = list(MINI_PARAMORPHIX_ALLOWED_MOVES.keys())
        generators = [MINI_PARAMORPHIX_ALLOWED_MOVES[k] for k in generator_names]
        initial_state = list(range(len(generators[0])))
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    else:
        raise ValueError(f"Unknown generator set: {name}")
