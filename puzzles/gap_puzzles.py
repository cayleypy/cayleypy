import json
import os
import re
from pathlib import Path
from typing import Optional, Union

from ..cayley_graph_def import CayleyGraphDef
from ..permutation_utils import permutation_from_cycles


def _cycle_str_to_list(cycle_str: str) -> list[list[int]]:
    """Converts a cycle string from GAP format to a list of integers.

    Example: "(1,2,3)(4,5)" -> [[1, 2, 3], [4, 5]]
    """
    return [list(map(int, group.split(","))) for group in re.findall(r"\(([\d,]+)\)", cycle_str)]


def _central_state_from_ip(n: int, ip: list[list[int]]) -> list[int]:
    """Given list of lists of identical pieces, generates central state.

    Pieces in `ip` are 1-indexed.
    Returns array `ans` of length n such that ans[i]=ans[j] if and only if pieces i and j are identical.
    Values in this array are integers from 0 to k-1 where k is number of colors.
    It is allowed to not specify singleton groups of equivalent pieces.
    """
    pos_to_eq_list = {}  # type: dict[int, list[int]]
    for eq_list in ip:
        for pos in eq_list:
            pos_to_eq_list[pos - 1] = eq_list

    ans = [-1] * n
    color = 0
    for i in range(n):
        if ans[i] != -1:
            continue
        if i in pos_to_eq_list:
            for j in pos_to_eq_list[i]:
                ans[j - 1] = color
        else:
            ans[i] = color
        color += 1
    return ans


def _parse_gap_file(text: str) -> CayleyGraphDef:
    generators_dict = {}  # type: dict[str, list[list[int]]]
    generator_names = []  # type: list[str]
    ip = None  # type: Optional[list[list[int]]]
    for line in text.split("\n"):
        if ":=" not in line:
            continue
        key, value = line.split(":=")
        value = value.replace(";", "")
        if key.startswith("M_"):
            gen_name = key.replace("M_", "")
            generator_names.append(gen_name)
            generators_dict[gen_name] = _cycle_str_to_list(value)
        elif key == "ip":
            ip = json.loads(value)

    n = max(max(cycle) for gen in generators_dict.values() for cycle in gen)
    generators = [permutation_from_cycles(n, generators_dict[gen_name], offset=1) for gen_name in generator_names]
    graph = CayleyGraphDef.create(generators=generators, generator_names=generator_names)
    if ip is not None:
        graph = graph.with_central_state(_central_state_from_ip(n, ip))
    return graph


class GapPuzzles:
    """Library of puzzles defined in GAP format."""

    @staticmethod
    def load_puzzle_from_file(file_name: Union[str, Path]) -> CayleyGraphDef:
        """Reads puzzle from given file in GAP format."""
        file_name = str(file_name)
        assert file_name.endswith(".gap"), "Must be .fap file."
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
        return _parse_gap_file(text)

    @staticmethod
    def get_gaps_dir() -> Path:
        """Path where all GAP files are stored.

        Only puzzles in "defaults" subfolder can be loaded with GapPuzzles.puzzle.
        Puzzles in other subfolders can be loaded with ``GapPuzzles.load_puzzle_from_file``.
        """
        return Path(__file__).parent / "gap_files"

    @staticmethod
    def list_puzzles() -> list[str]:
        """Lists puzzles that can be loaded with ``GapPuzzles.puzzle``."""
        gaps_dir = GapPuzzles.get_gaps_dir() / "defaults"
        return sorted([f.stem for f in gaps_dir.glob("*.gap")])

    @staticmethod
    def puzzle(puzzle_name: str, make_inverse_closed: bool = True) -> CayleyGraphDef:
        """Loads puzzle by name.

        :param puzzle_name: Puzzle name.
        :param make_inverse_closed: Whether to make graph inverse-closed (if it was not already).
        :return: CayleyGraphDef for this puzzle.
        """
        file_name = str(GapPuzzles.get_gaps_dir() / "defaults" / f"{puzzle_name}.gap")
        try:
            graph_def = GapPuzzles.load_puzzle_from_file(file_name)
        except FileNotFoundError as exc:
            raise ValueError(
                f"No such puzzle {puzzle_name}. Use GapPuzzles.list_puzzles() to see list of available puzzles."
            ) from exc

        graph_def = graph_def.with_name(f"{puzzle_name}.gap")
        if not graph_def.generators_inverse_closed and make_inverse_closed:
            graph_def = graph_def.make_inverse_closed()
        return graph_def


if os.getenv("SPHINX_BUILD") == "1":
    GapPuzzles.__doc__ = "%s\n\n Supported puzzles: %s." % (GapPuzzles.__doc__, GapPuzzles.list_puzzles())
