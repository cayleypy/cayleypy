import itertools
import os

import pytest

from cayleypy.permutation_utils import compose_permutations
from .hungarian_rings import (
    hungarian_rings_permutations,
    _circular_shift,
    _create_right_ring,
    _get_intersections,
    get_santa_parameters_from_n,
    get_group,
    get_pair_variants,
    hungarian_rings_generators,
)
from .. import CayleyGraphDef, CayleyGraph, bfs_numpy

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"

circular_shift_test_data = [
    (10, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    (10, 11, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    (10, 3, [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
    (10, 9, [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    (10, -1, [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    (10, -3, [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]),
    (10, -9, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    (10, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (1, 1, [0]),
    (0, 1, []),
]


@pytest.mark.parametrize("size, step, expected_result", circular_shift_test_data)
def test_circular_shift(size, step, expected_result):
    assert _circular_shift(items=list(range(0, size)), step=step) == expected_result


@pytest.mark.parametrize(
    "left_size, left_index, right_size, right_index, full_size, exp_result",
    [
        (6, 2, 6, 2, 10, [0, 6, 7, 8, 2, 9]),
        (6, 2, 5, 1, 9, [0, 6, 7, 8, 2]),
        (4, 0, 4, 0, 7, [0, 4, 5, 6]),
        (1, 0, 1, 0, 1, [0]),
    ],
)
def test_create_right_ring(left_size, left_index, right_size, right_index, full_size, exp_result):
    right_ring = _create_right_ring(left_size, left_index, right_size, right_index, full_size)
    assert right_ring == exp_result


@pytest.mark.parametrize(
    "left_index, right_index, exp_intersections",
    [
        (2, 2, 2),
        (2, 1, 2),
        (0, 0, 1),
        pytest.param(0, 1, None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(-1, 6, None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(2, -1, None, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_get_intersections(left_index, right_index, exp_intersections):
    intersections = _get_intersections(left_index=left_index, right_index=right_index)
    assert intersections == exp_intersections


hr_symmetric_test_data = [
    (6, 3, 6, 3, 1, [1, 2, 3, 4, 5, 0, 6, 7, 8, 9], [6, 1, 2, 8, 4, 5, 7, 3, 9, 0]),
    (6, 2, 6, 2, 1, [1, 2, 3, 4, 5, 0, 6, 7, 8, 9], [6, 1, 9, 3, 4, 5, 7, 8, 2, 0]),
    (6, 1, 6, 1, 1, [1, 2, 3, 4, 5, 0, 6, 7, 8, 9], [6, 0, 2, 3, 4, 5, 7, 8, 9, 1]),
    (6, 0, 6, 0, 1, [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10], [6, 1, 2, 3, 4, 5, 7, 8, 9, 10, 0]),
]

hr_test_data = [
    (6, 2, 3, 1, 1, [1, 2, 3, 4, 5, 0, 6], [6, 1, 0, 3, 4, 5, 2]),
    (6, 1, 2, 1, 1, [1, 2, 3, 4, 5, 0], [1, 0, 2, 3, 4, 5]),
    (6, 0, 1, 0, 1, [1, 2, 3, 4, 5, 0], [0, 1, 2, 3, 4, 5]),
    (1, 0, 6, 0, 1, [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]),
    (2, 1, 6, 1, 1, [1, 0, 2, 3, 4, 5], [2, 0, 3, 4, 5, 1]),
    (2, 1, 6, 2, 1, [1, 0, 2, 3, 4, 5], [2, 5, 3, 4, 1, 0]),
    (2, 0, 1, 0, 1, [1, 0], [0, 1]),
    (2, 0, 2, 0, 1, [1, 0, 2], [2, 1, 0]),
]

wreath_moves = [
    (6, 2, 6, 3, 1, [1, 2, 3, 4, 5, 0, 6, 7, 8, 9], [6, 1, 8, 3, 4, 5, 7, 2, 9, 0]),
    (7, 2, 7, 3, 1, [1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11], [7, 1, 10, 3, 4, 5, 6, 8, 9, 2, 11, 0]),
    (
        12,
        3,
        12,
        4,
        1,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [12, 1, 2, 19, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 3, 20, 21, 0],
    ),
]


@pytest.mark.parametrize(
    "left_size, left_index, right_size, right_index, step, expected_result_l, expected_result_r",
    hr_symmetric_test_data + hr_test_data + wreath_moves,
)
def test_hr_permutations(left_size, left_index, right_size, right_index, step, expected_result_l, expected_result_r):
    l_permutations, r_permutations = hungarian_rings_permutations(left_size, left_index, right_size, right_index, step)
    intersections = _get_intersections(left_index=left_index, right_index=right_index)

    full_size = left_size + right_size - intersections
    assert len(l_permutations) == len(r_permutations) == full_size
    assert l_permutations == expected_result_l
    assert r_permutations == expected_result_r


hr_value_error_data = [
    (0, 0, 1, 0, 1),
    (-1, 0, 1, 0, 1),
    (1, 0, 0, 0, 1),
    (1, 0, -1, 0, 1),
    (3, 3, 4, 0, 1),
    (3, 1, 4, 7, 1),
]


@pytest.mark.parametrize("left_size,left_index,right_size,right_index,step", hr_value_error_data)
def test_hr_permutations_value_errors(left_size: int, left_index: int, right_size: int, right_index: int, step: int):
    with pytest.raises(ValueError):
        hungarian_rings_permutations(left_size, left_index, right_size, right_index, step=step)


@pytest.mark.parametrize(
    "left_size, left_index, right_size, right_index, step", [(6, 3, 6, 2, 2), (6, 1, 5, 2, -7), (13, 9, 5, 2, 6)]
)
def test_hr_permutations_compensation(left_size: int, left_index: int, right_size: int, right_index: int, step: int):
    left_index = min(left_index, left_size - 1)
    right_index = min(right_index, right_size - 1)
    l_permutations, r_permutations = hungarian_rings_permutations(
        left_size, left_index, right_size, right_index, step=step
    )
    intersections = _get_intersections(left_index=left_index, right_index=right_index)
    full_size = left_size + right_size - intersections

    l_counter_perm, r_counter_perm = hungarian_rings_permutations(
        left_size, left_index, right_size, right_index, step=-step
    )

    assert len(l_permutations) == len(r_permutations) == full_size
    assert compose_permutations(l_permutations, l_counter_perm) == list(range(full_size))
    assert compose_permutations(r_permutations, r_counter_perm) == list(range(full_size))


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_hr_permutations_compensation_bf():
    two_inter_params = [range(2, 6), range(1, 5), range(2, 6), range(1, 5), range(-7, 8)]
    one_inter_params = [range(2, 6), [0], range(2, 6), [0], range(-7, 8)]
    parameters = list(itertools.product(*one_inter_params)) + list(itertools.product(*two_inter_params))
    for left_size, left_index, right_size, right_index, step in parameters:
        if left_index >= left_size or right_index >= right_size:
            continue
        l_permutations, r_permutations = hungarian_rings_permutations(
            left_size, left_index, right_size, right_index, step=step
        )
        intersections = _get_intersections(left_index=left_index, right_index=right_index)
        full_size = left_size + right_size - intersections

        l_counter_perm, r_counter_perm = hungarian_rings_permutations(
            left_size, left_index, right_size, right_index, step=-step
        )

        assert len(l_permutations) == len(r_permutations) == full_size
        assert compose_permutations(l_permutations, l_counter_perm) == list(range(full_size))
        assert compose_permutations(r_permutations, r_counter_perm) == list(range(full_size))


@pytest.mark.parametrize(
    "n, parameters",
    [
        (10, (6, 2, 6, 3)),
        (12, (7, 2, 7, 3)),
    ],
)
def test_get_santa_parameters_from_n(n: int, parameters):
    assert get_santa_parameters_from_n(n) == parameters


pairs_data = {
    (2, 4): [
        (2, 1, 4, 1),
        (2, 1, 4, 2),
    ],
    (2, 5): [
        (2, 1, 5, 1),
        (2, 1, 5, 2),
    ],
    (2, 6): [
        (2, 1, 6, 1),
        (2, 1, 6, 2),
        (2, 1, 6, 3),
    ],
    (3, 4): [
        (3, 1, 4, 1),
        (3, 1, 4, 2),
    ],
    (3, 5): [
        (3, 1, 5, 1),
        (3, 1, 5, 2),
    ],
    (4, 4): [
        (4, 1, 4, 1),
        (4, 1, 4, 2),
        (4, 2, 4, 2),
    ],
}

groups_data = [
    (4, [(2, 0, 3, 0)] + pairs_data[(2, 4)] + [(3, 1, 3, 1)]),
    (5, [(2, 0, 4, 0), (3, 0, 3, 0)] + pairs_data[(2, 5)] + pairs_data[(3, 4)]),
    (6, [(2, 0, 5, 0), (3, 0, 4, 0)] + pairs_data[(2, 6)] + pairs_data[(3, 5)] + pairs_data[(4, 4)]),
]


@pytest.mark.parametrize("pair, variants", pairs_data.items())
def test_get_pair_variants(pair: tuple[int, int], variants: list):
    assert get_pair_variants(*pair) == variants


@pytest.mark.parametrize("n, group", groups_data)
def test_get_group(n: int, group: list):
    assert get_group(n) == group


layer_sizes_data = [
    (
        (2, 1, 7, 3),
        [1, 3, 6, 12, 20, 34, 55, 83, 124, 185, 274, 395, 558, 726, 808, 739, 540, 323, 104, 26, 14, 6, 3, 1],
    ),
    (
        (7, 3, 2, 1),
        [1, 3, 6, 12, 20, 34, 55, 83, 124, 185, 274, 395, 558, 726, 808, 739, 540, 323, 104, 26, 14, 6, 3, 1],
    ),
    ((5, 1, 6, 1), [1, 4, 12, 33, 88, 232, 608, 1596, 4085, 10132, 24209, 53006, 95034, 111383, 56032, 6323, 101, 1]),
]


@pytest.mark.parametrize("parameters, layer_sizes", layer_sizes_data)
def test_layer_sizes(parameters: tuple[int, int, int, int], layer_sizes: list[int]):
    generators, generator_names = hungarian_rings_generators(*parameters)
    n = len(generators[0])
    graph_def = CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    assert bfs_numpy(CayleyGraph(graph_def)) == layer_sizes
    assert CayleyGraph(graph_def).bfs().layer_sizes == layer_sizes
