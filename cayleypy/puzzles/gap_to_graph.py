import re
from pathlib import Path
import numpy as np
import torch
from ..io_utils import read_txt
from ..permutation_utils import inverse_permutation, permutation_from_cycles, inverse_permutation
from ..cayley_graph_def import CayleyGraphDef


def perm_is_id(perm: list[int]) -> bool:
    return np.all(perm == np.arange(len(perm)))


def add_inv_permutations(perms_dict: dict[str, list[int]]) -> dict[str, list[int]]:
    """
    Combining original and inverse permutations
    WARNING: this function doesn't check if the inverted permutations are already present
    """
    perms_dict_all = {}

    for name, perm in perms_dict.items():
        if not perm_is_id(perm):
            perms_dict_all[name] = perm
            inv_perm = inverse_permutation(perm)
            if not np.all(inv_perm == perm):
                perms_dict_all[name + "_inv"] = np.array(inverse_permutation(perm))
    return perms_dict_all


def get_permuted_set_length(perms_cyclic: dict[list[list[int]]]) -> int:
    """
    Assuming that the first element is 0.
    """
    max_idx = 0
    for v in perms_cyclic.values():
        for cycle in v:
            for idx in cycle:
                if idx > max_idx:
                    max_idx = idx

    max_idx += 1  # because we start from 0
    return max_idx


def gap_lines_to_dict(moves_list_gap: list[str]) -> dict[str, str]:
    return_dict = {}
    for x in moves_list_gap:
        kv = x.replace(";", "").split(":=")
        return_dict[kv[0].replace("M_", "")] = kv[1]
    return return_dict


def filter_gap_generator_lines(gap_str: str) -> list[str]:
    return [x for x in gap_str.split("\n") if ":=" in x and "Gen" not in x and "ip" not in x]


def cycle_str_to_list(cycle_str: str, offset: int = 0) -> list[list[int]]:
    """
    Converts a cycle string from GAP format to a list of integers.
    Example: "(1,2,3)(4,5)" -> [[1, 2, 3], [4, 5]]
    """
    return [list(map(lambda x: int(x) - offset, group.split(","))) for group in re.findall(r"\(([\d,]+)\)", cycle_str)]


def gap_to_CayleyGraphDef(gap_file_path: str) -> CayleyGraphDef:
    gap_generators = read_txt(gap_file_path)
    gap_generators = filter_gap_generator_lines(gap_generators)
    gens_cyclic = gap_lines_to_dict(gap_generators)
    gens_cyclic = {k: cycle_str_to_list(v, 1) for k, v in gens_cyclic.items()}
    N = get_permuted_set_length(gens_cyclic)
    gens_oneline = {k: permutation_from_cycles(N, v) for k, v in gens_cyclic.items()}
    gens_oneline = add_inv_permutations(gens_oneline)
    names = []
    gens = []
    for k, v in gens_oneline.items():
        names.append(k)
        gens.append(v)
    return CayleyGraphDef.create(generators=gens, generator_names=names)


def get_gaps_dir() -> str:
    return Path(__file__).parent / "gap_files"


def list_gap_puzzles_defaults():
    gaps_dir = get_gaps_dir() / "defaults"
    return [f.stem for f in gaps_dir.glob("*.gap")]


def cayley_graph_for_puzzle_gap(puzzle_name: str) -> CayleyGraphDef:
    """
    Creates a CayleyGraphDef from a GAP file.
    """
    puzzle_name = puzzle_name.replace(" ", "_")
    list_of_puzzles = list_gap_puzzles_defaults()
    if puzzle_name not in list_of_puzzles:
        raise ValueError(
            f"Puzzle {puzzle_name} not found in the default GAP puzzles. Check list_gap_puzzles_defaults() for available puzzles."
        )
    gap_file_path = get_gaps_dir() / "defaults" / f"{puzzle_name}.gap"
    if not gap_file_path.exists():
        raise FileNotFoundError(f"GAP file {gap_file_path} does not exist.")
    return gap_to_CayleyGraphDef(gap_file_path)
