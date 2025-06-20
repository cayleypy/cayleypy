from typing import Dict
from cayleypy.permutation_utils import compose_permutations, apply_permutation, inverse_permutation
import collections
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

def generate_cube_permutations_oneline(n: int) -> Dict[str, list[int]]:
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

def full_set_of_perm_cube(cube_size: int) -> Dict[str, list[int]]:
    original_dict = generate_cube_permutations_oneline(cube_size)
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = list(map(int, value.split()))
        inv_key = key + '_inv'
        new_dict[inv_key] = inverse_permutation(list(map(int, value.split())))
    return new_dict

def get_cube_generators(cube_size: int, metric: str) -> Dict[str, list[int]]:
    if metric == "QTM":
        assert cube_size == 2 or cube_size == 3
        if cube_size==2:
            return CUBE222_ALLOWED_MOVES
        else:
            return CUBE333_ALLOWED_MOVES
    elif metric == "HTM":
        assert cube_size == 2 or cube_size == 3
        if cube_size==2:
            full_gens = CUBE222_ALLOWED_MOVES
            for move_id in ['f0', 'r1', 'd0']:
                full_gens[move_id + '^2'] = compose_permutations(CUBE222_ALLOWED_MOVES[move_id], CUBE222_ALLOWED_MOVES[move_id])
            return full_gens
        elif cube_size==3:
            full_gens = CUBE333_ALLOWED_MOVES
            for move_id in ['U', 'D', 'L', 'R', 'B', 'F']:
                full_gens[move_id + "^2"] = compose_permutations(CUBE333_ALLOWED_MOVES[move_id], CUBE333_ALLOWED_MOVES[move_id])
            return full_gens
    elif metric == "QSTM":
        return full_set_of_perm_cube(cube_size)
    else:
        raise ValueError(f"Unknown metric: {metric}")
