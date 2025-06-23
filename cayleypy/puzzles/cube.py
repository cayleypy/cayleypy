from typing import Dict

from cayleypy import CayleyGraphDef
from cayleypy.permutation_utils import compose_permutations, apply_permutation, inverse_permutation
import collections


def generate_cube_permutations_oneline(n: int) -> Dict[str, list[int]]:
    """
    Generates permutations for the basic moves of the n x n x n Rubik's cube.

    Arguments:
      n: The cube dimension (e.g. 3 for a 3x3x3 cube).

    Returns:
      A dictionary where keys are the names of the moves (e.g. 'f0', 'r1')
      and values are strings representing the permutations in single-line notation.
    """
    assert n >= 2, "n must be at least 2"
    faces = ["U", "F", "R", "B", "L", "D"]
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
    move_names_ordered = [f"{move_type}{i}" for move_type in ["f", "r", "d"] for i in range(n)]
    for move_name in move_names_ordered:
        move_type = move_name[0]
        s = int(move_name[1:])
        all_cycles = []
        if move_type == "f":
            for k in range(n):
                cycle = (
                    get_sticker_index("U", n - 1 - s, k),
                    get_sticker_index("R", k, s),
                    get_sticker_index("D", s, n - 1 - k),
                    get_sticker_index("L", n - 1 - k, n - 1 - s),
                )
                all_cycles.append(cycle[::-1])
        elif move_type == "r":
            side_cycles = []
            for k in range(n):
                s1 = get_sticker_index("U", k, s)
                s2 = get_sticker_index("F", k, s)
                s3 = get_sticker_index("D", k, s)
                s4 = get_sticker_index("B", n - 1 - k, n - 1 - s)
                side_cycles.append((s1, s2, s3, s4))
            all_cycles.extend(side_cycles)
            if s == n - 1:
                face_cycles_cw = rotate_face_cw("R")
                all_cycles.extend([c[::-1] for c in face_cycles_cw])
            if s == 0:
                face_cycles_cw = rotate_face_cw("L")
                all_cycles.extend(face_cycles_cw)
        elif move_type == "d":
            for k in range(n):
                cycle = (
                    get_sticker_index("F", n - 1 - s, k),
                    get_sticker_index("L", n - 1 - s, k),
                    get_sticker_index("B", n - 1 - s, k),
                    get_sticker_index("R", n - 1 - s, k),
                )
                all_cycles.append(cycle)
        face_to_rotate = None
        if move_type == "f" and s == 0:
            face_to_rotate = "F"
        if move_type == "f" and s == n - 1:
            face_to_rotate = "B"
        if move_type == "r" and s == 0:
            face_to_rotate = "L"
        if move_type == "r" and s == n - 1:
            face_to_rotate = "R"
        if move_type == "d" and s == 0:
            face_to_rotate = "D"
        if move_type == "d" and s == n - 1:
            face_to_rotate = "U"

        if face_to_rotate:
            face_cycles_cw = rotate_face_cw(face_to_rotate)
            is_ccw = False
            if (move_type == "f" or move_type == "d") and s == 0:
                is_ccw = True
            if move_type == "r" and s == n - 1:
                is_ccw = True
            if is_ccw:
                all_cycles.extend([c[::-1] for c in face_cycles_cw])
            else:
                all_cycles.extend(face_cycles_cw)
        p = list(range(total_stickers))
        for cycle in all_cycles:
            if len(cycle) < 2:
                continue
            for i in range(len(cycle)):
                p[cycle[i]] = cycle[(i + 1) % len(cycle)]
        output_move_name = move_name
        if move_type == "r":
            output_move_name = f"r{n-1-s}"
        moves[output_move_name] = " ".join(map(str, p))
    sorted_moves = collections.OrderedDict(sorted(moves.items(), key=lambda t: move_names_ordered.index(t[0])))
    return dict(sorted_moves)


def full_set_of_perm_cube(cube_size: int) -> Dict[str, list[int]]:
    original_dict = generate_cube_permutations_oneline(cube_size)
    new_dict = {}
    for key, value in original_dict.items():
        new_dict[key] = list(map(int, value.split()))
        inv_key = key + "_inv"
        new_dict[inv_key] = inverse_permutation(list(map(int, value.split())))
    return new_dict


def graph_for_cube(cube_size: int, metric: str) -> CayleyGraphDef:
    if metric == "QSTM":
        return full_set_of_perm_cube(cube_size)
    else:
        raise ValueError(f"Unknown metric: {metric}")
