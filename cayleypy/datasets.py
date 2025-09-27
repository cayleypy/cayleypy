"""Helpers for computing and loading pre-computed results."""

import csv
import functools
import json
import os
import math
from typing import Any, Callable

from .cayley_graph import CayleyGraph
from .graphs_lib import prepare_graph, PermutationGroups, MatrixGroups
from .puzzles.hungarian_rings import get_group as get_hr_group
from .puzzles.puzzles import Puzzles

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@functools.cache
def load_dataset(dataset_name: str, error_if_not_found=True) -> dict[str, Any]:
    """Loads named dataset."""
    file_name = os.path.join(DATA_DIR, dataset_name + ".csv")
    data: dict[str, str] = {}
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as csvfile:
            for key, value in csv.reader(csvfile):
                data[key] = json.loads(value)
    else:
        if error_if_not_found:
            raise KeyError(f"No such dataset: {dataset_name}")
    return data


def _update_dataset(
    dataset_name: str, keys: list[str], eval_func: Callable[[str], Any]
):
    file_name = os.path.join(DATA_DIR, dataset_name + ".csv")
    data = load_dataset(dataset_name, error_if_not_found=False)
    for key in keys:
        if key not in data:
            data[key] = json.dumps(eval_func(key))
    rows = list(data.items())
    if "coset" in dataset_name:
        rows.sort(key=lambda x: (len(x[0]), x[0]))
    else:
        rows.sort(key=lambda x: tuple(map(int, x[0].split(","))))
    with open(file_name, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
    print(f"Updated: {file_name}")


# The code below can be viewed as definition of what is stored in datasets.
# It is used to compute results for small graphs. Results for larger are computed separately and added to repository
# manually.
def _compute_lrx_coset_growth(central_state: str) -> list[int]:
    n = len(central_state)
    graph_def = PermutationGroups.lrx(n).with_central_state(central_state)
    return CayleyGraph(graph_def).bfs().layer_sizes


def _compute_top_spin_coset_growth(central_state: str) -> list[int]:
    n = len(central_state)
    graph_def = PermutationGroups.top_spin(n).with_central_state(central_state)
    return CayleyGraph(graph_def).bfs().layer_sizes


def _compute_lrx_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(prepare_graph("lrx", n=int(n))).bfs().layer_sizes


def _compute_lx_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.lx(int(n))).bfs().layer_sizes


def _compute_top_spin_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(prepare_graph("top_spin", n=int(n))).bfs().layer_sizes


@functools.cache
def _stirling(n, k):
    """Computes unsigned Stirling number of the first kind."""
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    return (n - 1) * _stirling(n - 1, k) + _stirling(n - 1, k - 1)


def _compute_all_transpositions_cayley_growth(n_str: str) -> list[int]:
    # Growth function is given by Stirling numbers, see https://oeis.org/A094638.
    n = int(n_str)
    return [_stirling(n, n + 1 - k) for k in range(1, n + 1)]


def _compute_transposons_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.transposons(int(n))).bfs().layer_sizes


def _compute_block_interchange_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.block_interchange(int(n))).bfs().layer_sizes


def _compute_pancake_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.pancake(int(n))).bfs().layer_sizes


def _compute_burnt_pancake_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.burnt_pancake(int(n))).bfs().layer_sizes


def _compute_full_reversals_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.full_reversals(int(n))).bfs().layer_sizes


def _compute_signed_reversals_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.signed_reversals(int(n))).bfs().layer_sizes


# For S_n with Coxeter generators (adjacent transpositions), the number of permutations of length k equals
# the number of permutations in S_n with exactly k inversions.
# This is the coefficient of q^k in the q-factorial [n]_q!
def _compute_coxeter_cayley_growth(n_str: str) -> list[int]:
    # dp[k] = number of permutations in S_n with exactly k inversions.
    # Using dynamic programming to compute Mahonian numbers.
    n = int(n_str)
    max_inv = math.comb(n, 2)
    dp = [0] * (max_inv + 1)
    dp[0] = 1  # 1 permutation of length 0 with 0 inversions

    for i in range(1, n):  # i is the number of elements inserted so far
        new_dp = [0] * (max_inv + 1)
        for k in range(max_inv + 1):
            for j in range(
                min(i + 1, k + 1)
            ):  # insert new element at position j (adds j inversions)
                new_dp[k] += dp[k - j]
        dp = new_dp

    return dp


def _compute_cyclic_coxeter_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.cyclic_coxeter(int(n))).bfs().layer_sizes


def _compute_hungarian_rings_growth(key: str) -> list[int]:
    parameters = map(int, key.split(","))
    return CayleyGraph(Puzzles.hungarian_rings(*parameters)).bfs().layer_sizes


def _compute_all_cycles_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.all_cycles(int(n))).bfs().layer_sizes


def _compute_heisenberg_growth(key: str) -> list[int]:
    n, modulo = map(int, key.split(","))
    return CayleyGraph(MatrixGroups.heisenberg(n=n, modulo=modulo)).bfs().layer_sizes


def _compute_sl_fund_roots_growth(n: str, m: str) -> list[int]:
    return (
        CayleyGraph(
            MatrixGroups.special_linear_fundamental_roots(int(n), modulo=int(m))
        )
        .bfs()
        .layer_sizes
    )


def _compute_sl_root_weyl_growth(n: str, m: str) -> list[int]:
    return (
        CayleyGraph(MatrixGroups.special_linear_root_weyl(int(n), modulo=int(m)))
        .bfs()
        .layer_sizes
    )


def _compute_rapaport_m1_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.rapaport_m1(int(n))).bfs().layer_sizes


def _compute_lsl_cycles_cayley_growth(n: int) -> list[int]:
    graph_def = PermutationGroups.lsl_cycles(n)
    return CayleyGraph(graph_def).bfs().layer_sizes


def _compute_rapaport_m2_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.rapaport_m2(int(n))).bfs().layer_sizes


def _compute_wrapped_k_cycles_cayley_growth(key: str) -> list[int]:
    n, k = map(int, key.split(","))
    return CayleyGraph(PermutationGroups.wrapped_k_cycles(n, k)).bfs().layer_sizes


def _compute_stars_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.stars(int(n))).bfs().layer_sizes


def _compute_larx_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.larx(int(n))).bfs().layer_sizes


def _compute_increasing_k_cycles_cayley_growth(key: str) -> list[int]:
    n, k = map(int, key.split(","))
    return CayleyGraph(PermutationGroups.increasing_k_cycles(n, k)).bfs().layer_sizes


def _compute_consecutive_k_cycles_cayley_growth(key: str) -> list[int]:
    n, k = map(int, key.split(","))
    return CayleyGraph(PermutationGroups.consecutive_k_cycles(n, k)).bfs().layer_sizes


def _compute_down_cycles_cayley_growth(n: str) -> list[int]:
    return CayleyGraph(PermutationGroups.down_cycles(int(n))).bfs().layer_sizes


def generate_datasets():
    """Generates datasets for small n, keeping existing values."""
    keys = []
    for n in range(3, 30):
        keys += ["01" * (n // 2) + "0" * (n % 2)]
        keys += ["0" * (n // 2 + n % 2) + "1" * (n // 2)]
    _update_dataset("lrx_coset_growth", keys, _compute_lrx_coset_growth)
    keys = [key for key in keys if len(key) >= 4]
    _update_dataset("top_spin_coset_growth", keys, _compute_top_spin_coset_growth)
    keys = [str(n) for n in range(3, 12)]
    _update_dataset("lrx_cayley_growth", keys, _compute_lrx_cayley_growth)
    _update_dataset("lx_cayley_growth", keys, _compute_lx_cayley_growth)
    keys = [str(n) for n in range(4, 12)]
    _update_dataset("top_spin_cayley_growth", keys, _compute_top_spin_cayley_growth)
    keys = [str(n) for n in range(2, 31)]
    _update_dataset(
        "all_transpositions_cayley_growth",
        keys,
        _compute_all_transpositions_cayley_growth,
    )
    _update_dataset("coxeter_cayley_growth", keys, _compute_coxeter_cayley_growth)
    keys = [str(n) for n in range(2, 11)]
    _update_dataset(
        "transposons_cayley_growth", keys, _compute_transposons_cayley_growth
    )
    _update_dataset("pancake_cayley_growth", keys, _compute_pancake_cayley_growth)
    _update_dataset(
        "full_reversals_cayley_growth", keys, _compute_full_reversals_cayley_growth
    )
    _update_dataset(
        "cyclic_coxeter_cayley_growth", keys, _compute_cyclic_coxeter_cayley_growth
    )
    _update_dataset(
        "rapaport_m1_cayley_growth", keys, _compute_rapaport_m1_cayley_growth
    )
    _update_dataset(
        "rapaport_m2_cayley_growth", keys, _compute_rapaport_m2_cayley_growth
    )
    _update_dataset(
        "down_cycles_cayley_growth", keys, _compute_down_cycles_cayley_growth
    )
    keys = [str(n) for n in range(1, 8)]
    _update_dataset(
        "burnt_pancake_cayley_growth", keys, _compute_burnt_pancake_cayley_growth
    )
    _update_dataset(
        "signed_reversals_cayley_growth", keys, _compute_signed_reversals_cayley_growth
    )
    keys = []
    for n in range(2, 10):
        group = get_hr_group(n)
        for parameters in group:
            keys.append(",".join([str(x) for x in parameters]))
    _update_dataset("hungarian_rings_growth", keys, _compute_hungarian_rings_growth)
    keys = [
        f"{n},{modulo}"
        for n in range(3, 11)
        for modulo in range(2, 51)
        if modulo ** (2 * n - 3) <= 2e6
    ]
    _update_dataset("heisenberg_growth", keys, _compute_heisenberg_growth)
    keys = [str(n) for n in range(2, 8)]
    _update_dataset("all_cycles_cayley_growth", keys, _compute_all_cycles_cayley_growth)
    keys = list(range(3, 11))
    _update_dataset("lsl_cycles_cayley_growth", keys, _compute_lsl_cycles_cayley_growth)
    keys = [str(n) for n in range(3, 10)]
    _update_dataset(
        "block_interchange_cayley_growth",
        keys,
        _compute_block_interchange_cayley_growth,
    )
    keys = [f"{n},{k}" for n in range(2, 10) for k in range(2, n + 1)]
    _update_dataset(
        "wrapped_k_cycles_cayley_growth", keys, _compute_wrapped_k_cycles_cayley_growth
    )
    keys = [str(n) for n in range(3, 12)]
    _update_dataset("stars_cayley_growth", keys, _compute_stars_cayley_growth)
    keys = [str(n) for n in range(2, 8)]
    _update_dataset("larx_cayley_growth", keys, _compute_larx_cayley_growth)
    keys = [str(n) for n in range(2, 11)]
    _update_dataset(
        "sl_2_fund_roots_growth", keys, lambda m: _compute_sl_fund_roots_growth(2, m)
    )
    keys = [str(n) for n in range(2, 6)]
    _update_dataset(
        "sl_3_fund_roots_growth", keys, lambda m: _compute_sl_fund_roots_growth(3, m)
    )
    keys = [str(n) for n in range(2, 11)]
    _update_dataset(
        "sl_2_root_weyl_growth", keys, lambda m: _compute_sl_root_weyl_growth(2, m)
    )
    keys = [str(n) for n in range(2, 6)]
    _update_dataset(
        "sl_3_root_weyl_growth", keys, lambda m: _compute_sl_root_weyl_growth(3, m)
    )
    keys = [f"{n},{k}" for n in range(3, 10) for k in range(2, min(n, 5) + 1)]
    _update_dataset(
        "increasing_k_cycles_cayley_growth",
        keys,
        _compute_increasing_k_cycles_cayley_growth,
    )
    keys = [f"{n},{k}" for n in range(3, 10) for k in range(2, min(n, 5) + 1)]
    _update_dataset(
        "consecutive_k_cycles_cayley_growth",
        keys,
        _compute_consecutive_k_cycles_cayley_growth,
    )
