"""Helpers for computing and loading pre-computed results."""
import csv
import functools
import json
import os
from typing import Any, Callable

from .cayley_graph import CayleyGraph
from .graphs_lib import prepare_graph

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@functools.cache
def load_dataset(dataset_name: str, error_if_not_found=True) -> dict[str, str]:
    """Loads named dataset."""
    file_name = os.path.join(DATA_DIR, dataset_name + '.csv')
    data: dict[str, str] = dict()
    if os.path.exists(file_name):
        with open(file_name, "r") as csvfile:
            for key, value in csv.reader(csvfile):
                data[key] = json.loads(value)
    else:
        if error_if_not_found:
            raise KeyError(f"No such dataset: {dataset_name}")
    return data


def _update_dataset(dataset_name: str, keys: list[str], eval_func: Callable[[str], Any]):
    file_name = os.path.join(DATA_DIR, dataset_name + '.csv')
    data = load_dataset(dataset_name, error_if_not_found=False)
    for key in keys:
        if key not in data:
            data[key] = json.dumps(eval_func(key))
    rows = [(key, value) for key, value in data.items()]
    rows.sort(key=lambda x: (len(x[0]), x[0]))
    with open(file_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
    print(f"Updated: {file_name}")


def _compute_lrx_coset_growth(initial_state: str) -> list[int]:
    n = len(initial_state)
    generators, _ = prepare_graph("lrx", n=n)
    result = CayleyGraph(generators, dest=initial_state).bfs()
    return result.layer_sizes


def _compute_top_spin_coset_growth(initial_state: str) -> list[int]:
    n = len(initial_state)
    generators, _ = prepare_graph("top_spin", n=n)
    result = CayleyGraph(generators, dest=initial_state).bfs()
    return result.layer_sizes


def _compute_lrx_cayley_growth(n: str) -> list[int]:
    generators, _ = prepare_graph("lrx", n=int(n))
    return CayleyGraph(generators).bfs().layer_sizes


def _compute_top_spin_cayley_growth(n: str) -> list[int]:
    generators, _ = prepare_graph("top_spin", n=int(n))
    return CayleyGraph(generators).bfs().layer_sizes


def generate_datasets():
    """Generates datasets for small n, keeping existing values."""
    keys = []
    for n in range(2, 26):
        keys += ["01" * (n // 2) + "0" * (n % 2)]
        keys += ["0" * (n // 2) + "1" * (n // 2 + n % 2)]
    _update_dataset("lrx_coset_growth", keys, _compute_lrx_coset_growth)
    keys = [key for key in keys if len(key) >= 4]
    _update_dataset("top_spin_coset_growth", keys, _compute_top_spin_coset_growth)

    keys = [str(n) for n in range(2, 12)]
    _update_dataset("lrx_cayley_growth", keys, _compute_lrx_cayley_growth)
    keys = [str(n) for n in range(4, 12)]
    _update_dataset("top_spin_cayley_growth", keys, _compute_top_spin_cayley_growth)
