import collections
from typing import Dict
import itertools

from ..cayley_graph import CayleyGraphDef
from ..permutation_utils import (
    compose_permutations,
    identity_perm,
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


def fixed_corner_cub_quarter() -> CayleyGraphDef:
    generators, generator_names = [], []
    for move_id, perm in CUBE222_MOVES.items():
        generators += [perm, inverse_permutation(perm)]
        generator_names += [move_id, move_id + "'"]
    central_state = [color for color in range(6) for _ in range(4)]
    return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)


def fixed_corner_cub_half() -> CayleyGraphDef:
    generators, generator_names = [], []
    for move_id, perm in CUBE222_MOVES.items():
        generators += [perm, inverse_permutation(perm), compose_permutations(perm, perm)]
        generator_names += [move_id, move_id + "'", move_id + "^2"]
    central_state = [color for color in range(6) for _ in range(4)]
    return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)


def generate_cube_permutations_oneline(n: int) -> Dict[str, str]:
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
            if move_type in ("f", "d") and s == 0:
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


def get_qtm_metric_moves(n: int) -> Dict[str, list[int]]:
    """
    Returns only QTM generators: quarter turns of all layers except central ones (if odd n).

    Arguments:
        n: Cube size (n x n x n).

    Returns:
        Dictionary with move names and their one-line permutations.
    """
    all_moves = generate_cube_permutations_oneline(n)
    allowed_moves = {}

    center_layer = (n - 1) // 2 if n % 2 == 1 else None

    for move_name, perm in all_moves.items():
        s = int(move_name[1:])

        # Skip central layers only if odd n
        if center_layer is not None and s == center_layer:
            continue

        perm_list = list(map(int, perm.split()))
        allowed_moves[move_name] = perm_list

    return allowed_moves


def get_htm_metric_moves(n: int) -> Dict[str, list[int]]:
    """
    Returns HTM generators: quarter and half turns of all layers except the central one (if odd n).

    Arguments:
        n: Cube size (n x n x n).

    Returns:
        Dictionary with move names (e.g., 'f0', 'f0_2') and one-line permutations.
    """
    all_moves = generate_cube_permutations_oneline(n)
    allowed_moves = {}

    center_layer = (n - 1) // 2 if n % 2 == 1 else None

    for move_name, perm in all_moves.items():
        s = int(move_name[1:])

        if center_layer is not None and s == center_layer:
            continue

        perm_list = list(map(int, perm.split()))
        allowed_moves[move_name] = perm_list
        perm_twice = [perm_list[perm_list[i]] for i in range(len(perm_list))]
        move_180_name = f"{move_name}^2"
        allowed_moves[move_180_name] = perm_twice

    return allowed_moves


def get_atm_metric_moves(n: int) -> Dict[str, list[int]]:
    """
    Generates all possible moves (generators) for an n x n x n cube in ATM metric.
    Arguments:
    n: The cube dimension.
    Returns:
    A dictionary where keys are descriptive names of moves (e.g. 'axis_Z_slice_0_cw')
    and values are permutations as lists of integers.
    """

    base_slice_perms_str = generate_cube_permutations_oneline(n)
    if not base_slice_perms_str:
        return {}
    total_stickers = 6 * n * n
    axis_slice_perms = {
        "X": [],  # r0, r1, ...
        "Y": [],  # r0, r1, ...
        "Z": [],  # r0, r1, ...
    }  # type: dict[str, list[dict[str, list[int]]]]
    slice_names_x = sorted([k for k in base_slice_perms_str if k.startswith("r")], key=lambda x: int(x[1:]))
    slice_names_y = sorted([k for k in base_slice_perms_str if k.startswith("d")], key=lambda x: int(x[1:]))
    slice_names_z = sorted([k for k in base_slice_perms_str if k.startswith("f")], key=lambda x: int(x[1:]))

    for name in slice_names_x:
        p_cw = [int(x) for x in base_slice_perms_str[name].split()]
        axis_slice_perms["X"].append({"cw": p_cw, "ccw": inverse_permutation(p_cw)})
    for name in slice_names_y:
        p_cw = [int(x) for x in base_slice_perms_str[name].split()]
        axis_slice_perms["Y"].append({"cw": p_cw, "ccw": inverse_permutation(p_cw)})
    for name in slice_names_z:
        p_cw = [int(x) for x in base_slice_perms_str[name].split()]
        axis_slice_perms["Z"].append({"cw": p_cw, "ccw": inverse_permutation(p_cw)})
    all_atm_moves = {}
    for axis_name, slices in axis_slice_perms.items():
        state_combinations = itertools.product([0, 1, 2], repeat=n)
        for combo in state_combinations:
            if all(s == 0 for s in combo):
                continue
            current_perm = identity_perm(total_stickers)
            move_name_parts = []

            for i, state in enumerate(combo):
                slice_name = f"s{i}"
                if state == 1:  # CW
                    current_perm = compose_permutations(slices[i]["cw"], current_perm)
                    move_name_parts.append(f"{slice_name}_cw")
                elif state == 2:  # CCW
                    current_perm = compose_permutations(slices[i]["ccw"], current_perm)
                    move_name_parts.append(f"{slice_name}_ccw")

            final_move_name = f"axis_{axis_name}_" + "_".join(move_name_parts)
            all_atm_moves[final_move_name] = current_perm

    return all_atm_moves


def rubik_cube_qstm(cube_size: int) -> CayleyGraphDef:
    """Creates Cayley graph for n*n*n Rubik's cube using the QSTM metric."""
    assert cube_size >= 2, "Cube size must be at least 2."
    generators = []
    generator_names = []
    moves = generate_cube_permutations_oneline(cube_size)
    for key, value in moves.items():
        perm = list(map(int, value.split()))
        generators += [perm, inverse_permutation(perm)]
        generator_names += [key, key + "_inv"]
    central_state = [color for color in range(6) for _ in range(cube_size**2)]
    return CayleyGraphDef.create(generators, generator_names, central_state)


def rubik_cube_qtm(cube_size: int) -> CayleyGraphDef:
    """Creates Cayley graph for n*n*n Rubik's cube using the QTM metric."""
    if cube_size < 2:
        raise ValueError(f"Metric QTM not supported for cube size {cube_size}.")
    generators, generator_names = [], []
    for move_id, perm in get_qtm_metric_moves(cube_size).items():
        generators += [perm, inverse_permutation(perm)]
        generator_names += [move_id, move_id + "'"]
    central_state = [color for color in range(6) for _ in range(cube_size**2)]
    return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)


def rubik_cube_htm(cube_size: int) -> CayleyGraphDef:
    """Creates Cayley graph for n*n*n Rubik's cube using the HTM metric."""
    if cube_size < 2:
        raise ValueError(f"Metric HTM not supported for cube size {cube_size}.")
    generators, generator_names = [], []
    for move_id, perm in get_htm_metric_moves(cube_size).items():
        if "^2" not in move_id:
            generators += [perm, inverse_permutation(perm)]
            generator_names += [move_id, move_id + "'"]
        else:
            generators += [perm]
            generator_names += [move_id]
    central_state = [color for color in range(6) for _ in range(cube_size**2)]
    return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)


def rubik_cube_atm(cube_size: int) -> CayleyGraphDef:
    """Creates Cayley graph for n*n*n Rubik's cube using the ATM(Axial Turn Metric) metric."""
    if cube_size < 2:
        raise ValueError(f"Metric ATM not supported for cube size {cube_size}.")
    generators, generator_names = [], []
    for move_id, perm in get_atm_metric_moves(cube_size).items():
        generators += [perm]
        generator_names += [move_id]
    central_state = [color for color in range(6) for _ in range(cube_size**2)]
    return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
