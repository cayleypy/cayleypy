"""Helpers for computing and loading pre-computed results."""

import csv
import functools
import json
import os
from typing import Any, Callable

from .cayley_graph import CayleyGraph
from .graphs_lib import prepare_graph

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


def _update_dataset(dataset_name: str, keys: list[str], eval_func: Callable[[str], Any]):
    file_name = os.path.join(DATA_DIR, dataset_name + ".csv")
    data = load_dataset(dataset_name, error_if_not_found=False)
    for key in keys:
        if key not in data:
            data[key] = json.dumps(eval_func(key))
    rows = list(data.items())
    rows.sort(key=lambda x: (len(x[0]), x[0]))
    with open(file_name, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
    print(f"Updated: {file_name}")


# The code below can be viewed as definition of what is stored in datasets.
# It is used to compute results for small graphs. Results for larger are computed separately and added to repository
# manually.
def _compute_lrx_coset_growth(initial_state: str) -> list[int]:
    n = len(initial_state)
    generators = prepare_graph("lrx", n=n).generators
    result = CayleyGraph(generators, dest=initial_state).bfs()
    return result.layer_sizes


def _compute_top_spin_coset_growth(initial_state: str) -> list[int]:
    n = len(initial_state)
    generators = prepare_graph("top_spin", n=n).generators
    result = CayleyGraph(generators, dest=initial_state).bfs()
    return result.layer_sizes


def _compute_lrx_cayley_growth(n: str) -> list[int]:
    return prepare_graph("lrx", n=int(n)).bfs().layer_sizes


def _compute_top_spin_cayley_growth(n: str) -> list[int]:
    return prepare_graph("top_spin", n=int(n)).bfs().layer_sizes


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


def _compute_pancake_cayley_growth(n: str) -> list[int]:
    return prepare_graph("pancake", n=int(n)).bfs().layer_sizes


def _compute_burnt_pancake_cayley_growth(n: str) -> list[int]:
    return prepare_graph("burnt_pancake", n=int(n)).bfs().layer_sizes


def _compute_full_reversals_cayley_growth(n: str) -> list[int]:
    return prepare_graph("full_reversals", n=int(n)).bfs().layer_sizes


def _compute_coxeter_cayley_growth(n: str) -> list[int]:
    return prepare_graph("coxeter", n=int(n)).bfs().layer_sizes


def _compute_mini_paramorphix_cayley_growth(_: str) -> list[int]:
    return prepare_graph("mini_paramorphix").bfs().layer_sizes


def _compute_cyclic_coxeter_cayley_growth(n: str) -> list[int]:
    return prepare_graph("cyclic_coxeter", n=int(n)).bfs().layer_sizes


def _compute_hungarian_rings_growth(n: str) -> list[int]:
    return prepare_graph("hungarian_rings", n=int(n)).bfs().layer_sizes


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
    keys = [str(n) for n in range(4, 12)]
    _update_dataset("top_spin_cayley_growth", keys, _compute_top_spin_cayley_growth)
    keys = [str(n) for n in range(2, 31)]
    _update_dataset("all_transpositions_cayley_growth", keys, _compute_all_transpositions_cayley_growth)
    keys = [str(n) for n in range(2, 11)]
    _update_dataset("pancake_cayley_growth", keys, _compute_pancake_cayley_growth)
    _update_dataset("full_reversals_cayley_growth", keys, _compute_full_reversals_cayley_growth)
    _update_dataset("coxeter_cayley_growth", keys, _compute_coxeter_cayley_growth)
    _update_dataset("cyclic_coxeter_cayley_growth", keys, _compute_cyclic_coxeter_cayley_growth)
    keys = [str(n) for n in range(1, 8)]
    _update_dataset("burnt_pancake_cayley_growth", keys, _compute_burnt_pancake_cayley_growth)
    _update_dataset("mini_paramorphix_cayley_growth", ["24"], _compute_mini_paramorphix_cayley_growth)
    keys = [str(n) for n in range(6, 14, 2)]
    _update_dataset("hungarian_rings_growth", keys, _compute_hungarian_rings_growth)
