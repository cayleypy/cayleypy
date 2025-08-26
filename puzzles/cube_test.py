import numpy as np

from .puzzles import Puzzles


def test_cube_222_qstm():
    graph = Puzzles.rubik_cube(2, metric="QSTM")
    assert graph.n_generators == 12
    exp_names = ["f0", "f0_inv", "f1", "f1_inv", "r0", "r0_inv", "r1", "r1_inv", "d0", "d0_inv", "d1", "d1_inv"]
    assert graph.generator_names == exp_names
    assert np.array_equal(
        graph.generators,
        [
            [0, 1, 19, 17, 6, 4, 7, 5, 2, 9, 3, 11, 12, 13, 14, 15, 16, 20, 18, 21, 10, 8, 22, 23],
            [0, 1, 8, 10, 5, 7, 4, 6, 21, 9, 20, 11, 12, 13, 14, 15, 16, 3, 18, 2, 17, 19, 22, 23],
            [18, 16, 2, 3, 4, 5, 6, 7, 8, 0, 10, 1, 13, 15, 12, 14, 22, 17, 23, 19, 20, 21, 11, 9],
            [9, 11, 2, 3, 4, 5, 6, 7, 8, 23, 10, 22, 14, 12, 15, 13, 1, 17, 0, 19, 20, 21, 16, 18],
            [0, 5, 2, 7, 4, 21, 6, 23, 10, 8, 11, 9, 3, 13, 1, 15, 16, 17, 18, 19, 20, 14, 22, 12],
            [0, 14, 2, 12, 4, 1, 6, 3, 9, 11, 8, 10, 23, 13, 21, 15, 16, 17, 18, 19, 20, 5, 22, 7],
            [4, 1, 6, 3, 20, 5, 22, 7, 8, 9, 10, 11, 12, 2, 14, 0, 17, 19, 16, 18, 15, 21, 13, 23],
            [15, 1, 13, 3, 0, 5, 2, 7, 8, 9, 10, 11, 12, 22, 14, 20, 18, 16, 19, 17, 4, 21, 6, 23],
            [0, 1, 2, 3, 4, 5, 18, 19, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 22, 20, 23, 21],
            [0, 1, 2, 3, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 6, 7, 21, 23, 20, 22],
            [1, 3, 0, 2, 16, 17, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 20, 21, 22, 23],
            [2, 0, 3, 1, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 4, 5, 18, 19, 20, 21, 22, 23],
        ],
    )


def test_cube_222_qtm():
    graph = Puzzles.rubik_cube(2, metric="QTM")
    assert graph.n_generators == 12


def test_cube_222_htm():
    graph = Puzzles.rubik_cube(2, metric="HTM")
    assert graph.n_generators == 18


def test_cube_333_qtm():
    graph = Puzzles.rubik_cube(3, metric="QTM")
    assert graph.n_generators == 12


def test_cube_333_htm():
    graph = Puzzles.rubik_cube(3, metric="HTM")
    assert graph.n_generators == 18


def test_cube_222_atm():
    graph = Puzzles.rubik_cube(2, metric="ATM")
    assert graph.n_generators == 24


def test_cube_333_atm():
    graph = Puzzles.rubik_cube(3, metric="ATM")
    assert graph.n_generators == 78
