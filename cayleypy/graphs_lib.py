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


def _create_coxeter_generators(n: int) -> list[list[int]]:
    gens = []
    for k in range(n - 1):
        perm = list(range(n))
        perm[k], perm[k + 1] = perm[k + 1], perm[k]
        gens.append(perm)
    return gens

import collections

def inverse_permutation(perm):
    # Create an empty list to hold the inverse permutation
    inverse = [0] * len(perm)
    
    # Iterate over the original permutation
    for i, p in enumerate(perm):
        # Place the index at the correct position in the inverse permutation
        inverse[p] = i
    
    return inverse

def generate_cube_permutations_oneline(n: int):
    """
    Generates permutations for the basic moves of the n x n x n Rubik's cube.

    Arguments:
      n: The cube dimension (e.g. 3 for a 3x3x3 cube).

    Returns:
      A dictionary where keys are the names of the moves (e.g. 'f0', 'r1')
      and values ​​are strings representing the permutations in single-line notation.
    """
    if n < 2:
        print("Wrong n")
        return

    faces = ['U', 'F', 'R', 'B', 'L', 'D']
    face_map = {name: i for i, name in enumerate(faces)}
    n_squared = n * n
    total_stickers = 6 * n_squared

    def get_sticker_index(face_name, row, col):
        face_idx = face_map[face_name]
        return face_idx * n_squared + row * n + col

    def rotate_face_cw(face_name):
        cycles = []
        permuted = [False] * n_squared
        for r_start in range(n):
            for c_start in range(n):
                if permuted[r_start * n + c_start]:
                    continue
                cycle = []
                r, c = r_start, c_start
                for _ in range(4):
                    sticker_idx = get_sticker_index(face_name, r, c)
                    if sticker_idx not in cycle:
                        cycle.append(sticker_idx)
                    permuted[r * n + c] = True
                    r, c = c, n - 1 - r
                if len(cycle) > 1:
                    cycles.append(tuple(cycle))
        return cycles
    moves = collections.OrderedDict()
    move_names_ordered = [f'{move_type}{i}' for move_type in ['f', 'r', 'd'] for i in range(n)]
    for move_name in move_names_ordered:
        move_type = move_name[0]
        s = int(move_name[1:])
        all_cycles = []
        if move_type == 'f':
            for k in range(n):
                cycle = (
                    get_sticker_index('U', n - 1 - s, k),
                    get_sticker_index('R', k, s),
                    get_sticker_index('D', s, n - 1 - k),
                    get_sticker_index('L', n - 1 - k, n - 1 - s)
                )
                all_cycles.append(cycle[::-1])
        elif move_type == 'r':
            side_cycles = []
            for k in range(n):
                s1 = get_sticker_index('U', k, s)
                s2 = get_sticker_index('F', k, s)
                s3 = get_sticker_index('D', k, s)
                s4 = get_sticker_index('B', n - 1 - k, n - 1 - s)
                side_cycles.append((s1, s2, s3, s4))
            all_cycles.extend(side_cycles)
            if s == n - 1:
                face_cycles_cw = rotate_face_cw('R')
                all_cycles.extend([c[::-1] for c in face_cycles_cw])
            if s == 0:
                face_cycles_cw = rotate_face_cw('L')
                all_cycles.extend(face_cycles_cw)
        elif move_type == 'd':
            for k in range(n):
                cycle = (
                    get_sticker_index('F', n - 1 - s, k),
                    get_sticker_index('L', n - 1 - s, k),
                    get_sticker_index('B', n - 1 - s, k),
                    get_sticker_index('R', n - 1 - s, k)
                )
                all_cycles.append(cycle)
        face_to_rotate = None
        if move_type == 'f' and s == 0: face_to_rotate = 'F'
        if move_type == 'f' and s == n - 1: face_to_rotate = 'B'
        if move_type == 'r' and s == 0: face_to_rotate = 'L'
        if move_type == 'r' and s == n - 1: face_to_rotate = 'R'
        if move_type == 'd' and s == 0: face_to_rotate = 'D'
        if move_type == 'd' and s == n - 1: face_to_rotate = 'U'

        if face_to_rotate:
            face_cycles_cw = rotate_face_cw(face_to_rotate)
            is_ccw = False
            if (move_type == 'f' or move_type == 'd') and s == 0:
                is_ccw = True
            if move_type == 'r' and s == n - 1:
                is_ccw = True
            if is_ccw:
                all_cycles.extend([c[::-1] for c in face_cycles_cw])
            else:
                all_cycles.extend(face_cycles_cw)
        p = list(range(total_stickers))
        for cycle in all_cycles:
            if len(cycle) < 2: continue
            for i in range(len(cycle)):
                p[cycle[i]] = cycle[(i + 1) % len(cycle)]
        output_move_name = move_name
        if move_type == 'r':
            output_move_name = f'r{n-1-s}'
        moves[output_move_name] = ' '.join(map(str, p))
    sorted_moves = collections.OrderedDict(sorted(moves.items(), key=lambda t: move_names_ordered.index(t[0])))
    return dict(sorted_moves)

def help_cyclic(start_pos, finish_pos, N):
    lst = []
    for i in range(start_pos):
        lst.append(i)
    for i in range(start_pos, finish_pos+1):
        lst.append((i+1) if i != finish_pos else start_pos)
    for i in range(finish_pos+1, N):
        lst.append(i)
    return lst


def globe_gens(A, B):
    gens = {}
    x_count = 2 * B
    y_count = A + 1
    N = 2 * (A + 1) * B
    for r_count in range(y_count):
        gens[f'r{r_count}'] = help_cyclic(r_count * x_count, (r_count + 1) * x_count - 1, N)

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
        gens[f'f{f_count}'] = lst

    return gens

def full_set_of_perm_cube(cube_size):
    original_dict = generate_cube_permutations_oneline(cube_size)
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = list(map(int, value.split()))
        inv_key = key + '_inv'
        new_dict[inv_key] = inverse_permutation(list(map(int, value.split())))
    return new_dict

def full_set_of_perm_globe(A, B):
    original_dict = globe_gens(A, B)
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = value
        if 'r' in key:
            inv_key = key + '_inv'
            new_dict[inv_key] = inverse_permutation(value)
    return new_dict
    

def prepare_graph(name, **kwargs) -> CayleyGraph:
    """Returns pre-defined Cayley or Schreier coset graph.

    Supported graphs:
      * "all_transpositions" - Cayley graph for S_n (n>=2), generated by all n(n-1)/2 transpositions.
      * "pancake" - Cayley graph for S_n (n>=2), generated by reverses of all prefixes. It has n-1 generators denoted
          R1,R2..R(n-1), where Ri is reverse of elements 0,1..i. See https://en.wikipedia.org/wiki/Pancake_graph.
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
      * "cube_n/n/n_gensQSTM" - QSTM(short for Quarter Slice Turn Metric),
          is a move count metric for the 3x3x3 in which any clockwise or counterclockwise 90-degree turn of any layer counts as one turn,
          and rotations do not count as moves.
      * "globeA/B" - Globe puzzle group, A + 1 cycle and 2B order 2 generators

    :param name: name of pre-defined graph.
    :param n: parameter (if applicable).
    :return: requested graph as `CayleyGraph`.
    """
    PARAM_REQUIREMENTS = {
        "all_transpositions": ["n"],
        "pancake": ["n"],
        "full_reversals": ["n"],
        "lrx": ["n"],
        "top_spin": ["n"],
        "coxeter": ["n"],
        "cube_n/n/n_gensQSTM": ["n"],
        "globeA/B": ["A", "B"],
        
    }
    required_params = PARAM_REQUIREMENTS.get(name, [])
    for param in required_params:
        if param not in kwargs:
            raise ValueError(f"Параметр '{param}' обязателен для графа типа '{name}'")
    params = {k: v for k, v in kwargs.items() if k in required_params}
    
    if name == "all_transpositions":
        n = params['n']
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
        n = params['n']
        assert n >= 2
        generators = []
        generator_names = []
        for prefix_len in range(2, n + 1):
            perm = list(range(prefix_len - 1, -1, -1)) + list(range(prefix_len, n))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len - 1))
        return CayleyGraph(generators, dest=list(range(n)), generator_names=generator_names)
    elif name == "full_reversals":
        n = params['n']
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
        n = params['n']
        assert n >= 3
        generators = [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [1, 0] + list(range(2, n))]
        generator_names = ["L", "R", "X"]
        return CayleyGraph(generators, dest=list(range(n)), generator_names=generator_names)
    elif name == "top_spin":
        n = params['n']
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
        n = params['n']
        assert n >= 2
        generators = _create_coxeter_generators(n)
        generator_names = [f"({i},{i+1})" for i in range(n-1)]
        initial_state = list(range(n))
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "cube_n/n/n_gensQSTM":
        n = params['n']
        assert n >= 2
        generators = list(full_set_of_perm_cube(n).values())
        generator_names = list(full_set_of_perm_cube(n).keys())
        initial_state = list(range(6*n**2))
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    elif name == "globeA/B":
        A = params['A']
        B = params['B']
        assert A >= 1
        assert B >= 1
        generators = list(full_set_of_perm_globe(A, B).values())
        generator_names = list(full_set_of_perm_globe(A, B).keys())
        initial_state = list(range(2*B*(A+1)))
        return CayleyGraph(generators, dest=initial_state, generator_names=generator_names)
    else:
        raise ValueError(f"Unknown generator set: {name}")
