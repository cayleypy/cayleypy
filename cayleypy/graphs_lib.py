# pylint: disable=line-too-long

"""Library of pre-defined graphs."""

from itertools import permutations

from cayleypy.cayley_graph import CayleyGraphDef
from cayleypy.hungarian_rings import hungarian_rings_generators
from cayleypy.permutation_utils import (
    compose_permutations,
    transposition,
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

MINI_PYRAMORPHIX_ALLOWED_MOVES = {
    "M_DF": [0, 1, 2, 3, 4, 5, 11, 9, 10, 7, 8, 6, 15, 16, 17, 12, 13, 14, 18, 19, 20, 21, 22, 23],
    "M_RL": [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 18, 19, 20],
    "M_DFv": [3, 4, 5, 0, 1, 2, 11, 9, 10, 7, 8, 6, 15, 16, 17, 12, 13, 14, 21, 22, 23, 18, 19, 20],
    "M_LF": [8, 6, 7, 3, 4, 5, 1, 2, 0, 9, 10, 11, 18, 19, 20, 15, 16, 17, 12, 13, 14, 21, 22, 23],
    "M_RD": [0, 1, 2, 10, 11, 9, 6, 7, 8, 5, 3, 4, 12, 13, 14, 21, 22, 23, 18, 19, 20, 15, 16, 17],
    "M_LFv": [8, 6, 7, 10, 11, 9, 1, 2, 0, 5, 3, 4, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17],
    "M_DL": [0, 1, 2, 8, 6, 7, 4, 5, 3, 9, 10, 11, 12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23],
    "M_FR": [10, 11, 9, 3, 4, 5, 6, 7, 8, 2, 0, 1, 21, 22, 23, 15, 16, 17, 18, 19, 20, 12, 13, 14],
    "M_DLv": [10, 11, 9, 8, 6, 7, 4, 5, 3, 2, 0, 1, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14],
    "M_Fv": [7, 8, 6, 5, 3, 4, 10, 11, 9, 1, 2, 0, 13, 14, 12, 22, 23, 21, 16, 17, 15, 19, 20, 18],
    "M_Fvi": [11, 9, 10, 4, 5, 3, 2, 0, 1, 8, 6, 7, 14, 12, 13, 20, 18, 19, 23, 21, 22, 17, 15, 16],
    "M_Dv": [2, 0, 1, 9, 10, 11, 3, 4, 5, 6, 7, 8, 19, 20, 18, 16, 17, 15, 22, 23, 21, 13, 14, 12],
    "M_Dvi": [1, 2, 0, 6, 7, 8, 9, 10, 11, 3, 4, 5, 23, 21, 22, 17, 15, 16, 14, 12, 13, 20, 18, 19],
    "M_Lv": [5, 3, 4, 7, 8, 6, 0, 1, 2, 11, 9, 10, 22, 23, 21, 13, 14, 12, 19, 20, 18, 16, 17, 15],
    "M_Lvi": [6, 7, 8, 1, 2, 0, 5, 3, 4, 10, 11, 9, 17, 15, 16, 23, 21, 22, 20, 18, 19, 14, 12, 13],
    "M_Rv": [9, 10, 11, 2, 0, 1, 8, 6, 7, 4, 5, 3, 16, 17, 15, 19, 20, 18, 13, 14, 12, 22, 23, 21],
    "M_Rvi": [4, 5, 3, 11, 9, 10, 7, 8, 6, 0, 1, 2, 20, 18, 19, 14, 12, 13, 17, 15, 16, 23, 21, 22],
}

# fmt: off
PYRAMINX_MOVES = {
    "r": pfc(36, [[31, 33, 32]], offset=1),
    "l": pfc(36, [[34, 36, 35]], offset=1),
    "b": pfc(36, [[28, 30, 29]], offset=1),
    "u": pfc(36, [[25, 27, 26]], offset=1),
    "BL": pfc(36, [[1, 8, 4], [2, 7, 3], [13, 22, 18], [14, 23, 16], [15, 24, 17], [25, 34, 30], [26, 35, 28], [27, 36, 29]], offset=1),
    "BR": pfc(36, [[3, 11, 10], [4, 12, 9], [13, 17, 19], [14, 18, 20], [15, 16, 21], [25, 29, 31], [26, 30, 32], [27, 28, 33]], offset=1),
    "F": pfc(36, [[1, 10, 5], [2, 9, 6], [13, 21, 23], [14, 19, 24], [15, 20, 22], [25, 33, 35], [26, 31, 36], [27, 32, 34]], offset=1),
    "D": pfc(36, [[5, 12, 7], [6, 11, 8], [16, 22, 19], [17, 23, 20], [18, 24, 21], [28, 34, 31], [29, 35, 32], [30, 36, 33]], offset=1),
}

STARMINX_MOVES = {
    'M_UFR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 38, 39, 40, 41, 42, 43, 47, 46, 37, 36, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 104, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_DIBF': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 51, 50, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 25, 24, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 38, 39, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 98, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_ULF': [0, 1, 2, 3, 4, 5, 6, 7, 31, 30, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 45, 44, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 8, 9, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 86, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_DBFE': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 20, 50, 51, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 19, 18, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 101, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_AFL': [0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 28, 29, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 9, 8, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 74, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_BREBF': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 57, 56, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 58, 20, 21, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 83, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_UBLL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 31, 30, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 35, 34, 32, 33, 12, 13, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_DEC': [0, 1, 2, 3, 4, 5, 19, 18, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 41, 40, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 6, 7, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_ILBL': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 34, 35, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 15, 14, 33, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 80, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_RCE': [0, 1, 43, 42, 4, 5, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 7, 6, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 119, 117, 118],
    'M_UBRBL': [13, 12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 55, 54, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 0, 1, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_DCA': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 53, 52, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 27, 26, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 40, 41, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 113, 111, 112, 114, 115, 116, 117, 118, 119],
    'M_BFBLBR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 58, 59, 50, 51, 52, 53, 48, 49, 56, 57, 54, 55, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 116, 114, 115, 117, 118, 119],
    'M_FAC': [0, 1, 2, 3, 52, 53, 6, 7, 8, 9, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 10, 11, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 68, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_URBR': [16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 47, 46, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 1, 0, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_DAI': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 23, 22, 25, 24, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 92, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_CRF': [0, 1, 5, 4, 36, 37, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 3, 2, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_BLBFI': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 39, 38, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 49, 48, 40, 41, 42, 43, 44, 45, 46, 47, 14, 15, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_EBRR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 56, 57, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 17, 16, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 43, 42, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'M_LIA': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 33, 32, 24, 25, 26, 27, 22, 23, 30, 31, 29, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 89, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
}
# fmt: on


def _create_coxeter_generators(n: int) -> list[list[int]]:
    return [transposition(n, k, k + 1) for k in range(n - 1)]


class PermutationGroups:
    """Pre-defined Cayley graphs for permutation groups (S_n)."""

    @staticmethod
    def all_transpositions(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by all n(n-1)/2 transpositions."""
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                generators.append(transposition(n, i, j))
                generator_names.append(f"({i},{j})")
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def full_reversals(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by reverses of all possible n(n-1)/2 substrings."""
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                perm = list(range(i)) + list(range(j, i - 1, -1)) + list(range(j + 1, n))
                generators.append(perm)
                generator_names.append(f"R[{i}..{j}]")
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def lrx(n: int, k: int = 1) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=3), generated by: shift left, shift right, swap two elements 0 and k.

        :param n: Size of permutations.
        :param k: Specifies that X is transposition of elements 0 and k. 1<=k<n.
            By default, k=1, which means X is transposition of first 2 elements.
        """
        assert n >= 3
        generators = [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), transposition(n, 0, k)]
        generator_names = ["L", "R", "X"]
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def top_spin(n: int, k: int = 4):
        """Cayley graph for S_n (n>=k>=2), generated by: shift left, shift right, reverse first k elements.

        :param n: Size of permutations.
        :param k: Specifies size of prefix to reverse. By default, k=4.
        """
        assert n >= k >= 2
        generators = [
            list(range(1, n)) + [0],
            [n - 1] + list(range(0, n - 1)),
            list(range(k - 1, -1, -1)) + list(range(k, n)),
        ]
        return CayleyGraphDef.create(generators, central_state=list(range(n)))

    @staticmethod
    def coxeter(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by adjacent transpositions (Coxeter generators).

        It has n-1 generators: (0,1), (1,2), ..., (n-2,n-1).
        """
        assert n >= 2
        generators = _create_coxeter_generators(n)
        generator_names = [f"({i},{i + 1})" for i in range(n - 1)]
        central_state = list(range(n))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)

    @staticmethod
    def cyclic_coxeter(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by adjacent transpositions plus cyclic transposition.

        It has n generators: (0,1), (1,2), ..., (n-2,n-1), (0,n-1).
        """
        assert n >= 2
        generators = _create_coxeter_generators(n) + [transposition(n, 0, n - 1)]
        generator_names = [f"({i},{i + 1})" for i in range(n - 1)] + [f"(0,{n - 1})"]
        central_state = list(range(n))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)

    @staticmethod
    def pancake(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by reverses of all prefixes.

        It has n-1 generators denoted R1,R2..R(n-1), where Ri is reverse of elements 0,1..i.
        See https://en.wikipedia.org/wiki/Pancake_graph.
        """
        assert n >= 2
        generators = []
        generator_names = []
        for prefix_len in range(2, n + 1):
            perm = list(range(prefix_len - 1, -1, -1)) + list(range(prefix_len, n))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len - 1))
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def burnt_pancake(n: int) -> CayleyGraphDef:
        """Cayley graph generated by reverses of all signed prefixes.

        Actually is a graph for S_2n (n>=1) representing a graph for n pancakes, where i-th element represents bottom
        side of i-th pancake, and (n+i)-th element represents top side of i-th pancake. The graph has n generators
        denoted R1,R2..R(n), where Ri is reverse of elements 0,1..i,n,n+1..n+i."""
        assert n >= 1
        generators = []
        generator_names = []
        for prefix_len in range(0, n):
            perm = []
            perm += list(range(n + prefix_len, n - 1, -1))
            perm += list(range(prefix_len + 1, n, 1))
            perm += list(range(prefix_len, -1, -1))
            perm += list(range(n + prefix_len + 1, 2 * n, 1))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len + 1))
        return CayleyGraphDef.create(generators, central_state=list(range(2 * n)), generator_names=generator_names)

    @staticmethod
    def three_cycles(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n ≥ 3), generated by all 3-cycles (a, b, c) where a < b, a < c."""
        assert n >= 3
        generators = []
        generator_names = []
        for a, b, c in permutations(range(n), 3):
            if a < b and a < c:
                generators.append(pfc(n, [[a, b, c]]))
                generator_names.append(f"({a} {b} {c})")
        return CayleyGraphDef(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def three_cycles_0ij(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n ≥ 3), generated by 3-cycles of the form (0 i j), where i != j."""
        generators = []
        generator_names = []
        for i, j in permutations(range(1, n), 2):
            generators.append(pfc(n, [[0, i, j]]))
            generator_names.append(f"({0} {i} {j})")
        return CayleyGraphDef(generators, central_state=list(range(n)), generator_names=generator_names)


def prepare_graph(name: str, n: int = 0) -> CayleyGraphDef:
    """Returns pre-defined Cayley or Schreier coset graph.

    TODO: move everything out and remove this function.
    DO NOT ADD ANY NEW GRAPHS HERE!

    Supported graphs:
      * "cube_2/2/2_6gensQTM" - Schreier coset graph for 2x2x2 Rubik's cube with fixed back left upper corner and only
          quarter-turns allowed. There are 6 generators (front, right, down face - clockwise and counterclockwise).
      * "cube_2/2/2_9gensHTM" - same as above, but allowing half-turns (it has 9 generators).
      * "cube_3/3/3_12gensQTM" - Schreier coset graph for 3x3x3 Rubik's cube with fixed central pieces and only
          quarter-turns allowed. There are 12 generators (clockwise and counterclockwise rotation for each face).
      * "cube_3/3/3_18gensHTM" - same as above, but allowing half-turns (it has 18 generators).
      * "mini_pyramorphix" – Cayley graph for a subgroup of S_24, acting on 24 titles. It is generated by 17 moves
          inspired by a simplified version of the Pyramorphix puzzle. Moves are based on overlapping 2- and 3-cycles
          and result in a symmetric, undirected graph. (order of the graph 24, degree 17, order of the group 288)
      * "pyraminx" - Cayley graph for a subgroup pf S_36 acting on 36 tiles. It is generated by 8 elements inspired
          by Pyraminx puzzle. 4 elemets represent rotations  of the tetrahedron tips, while 4 others -- rotations
          of its base layers.
      * "hungarian_rings" - Cayley graph for S_n (n>=4), generated by rotating two rings in both directions.
          For each ring structure and their intersection it has four generators.
      * "starminx" - Cayley graph, generated by 20 moves of Starminx puzzle. Each move is a rotation of a
          corner, centered around one of the dodecahedron’s pentagonal faces (face centres never move). A single turn
          corresponds to a set of disjoint 3-cycles, each on one triangular sticker.

    :param name: name of pre-defined graph.
    :param n: parameter (if applicable).
    :return: requested graph as `CayleyGraph`.
    """
    if name == "cube_2/2/2_6gensQTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE222_MOVES.items():
            generators += [perm, inverse_permutation(perm)]
            generator_names += [move_id, move_id + "'"]
        central_state = [color for color in range(6) for _ in range(4)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cube_2/2/2_9gensHTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE222_MOVES.items():
            generators += [perm, inverse_permutation(perm), compose_permutations(perm, perm)]
            generator_names += [move_id, move_id + "'", move_id + "^2"]
        central_state = [color for color in range(6) for _ in range(4)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cube_3/3/3_12gensQTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE333_MOVES.items():
            generators += [perm, inverse_permutation(perm)]
            generator_names += [move_id, move_id + "'"]
        central_state = [color for color in range(6) for _ in range(9)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "cube_3/3/3_18gensHTM":
        generators, generator_names = [], []
        for move_id, perm in CUBE333_MOVES.items():
            generators += [perm, inverse_permutation(perm), compose_permutations(perm, perm)]
            generator_names += [move_id, move_id + "'", move_id + "^2"]
        central_state = [color for color in range(6) for _ in range(9)]
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "mini_pyramorphix":
        generator_names = list(MINI_PYRAMORPHIX_ALLOWED_MOVES.keys())
        generators = [MINI_PYRAMORPHIX_ALLOWED_MOVES[k] for k in generator_names]
        central_state = list(range(len(generators[0])))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    elif name == "pyraminx":
        generator_names = []
        generators = []
        for move_id, perm in PYRAMINX_MOVES.items():
            generators += [perm, inverse_permutation(perm)]
            generator_names += [move_id, move_id + "_inv"]
        central_state = list(range(len(generators[0])))
        return CayleyGraphDef(generators, central_state=central_state, generator_names=generator_names)
    elif name == "hungarian_rings":
        assert n % 2 == 0
        ring_size = (n + 2) // 2
        assert ring_size >= 4
        generators, generator_names = hungarian_rings_generators(ring_size=ring_size)
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)
    elif name == "starminx":
        generator_names = list(STARMINX_MOVES.keys())
        generators = [STARMINX_MOVES[k] for k in generator_names]
        inversed_names = [f"inv_{k}" for k in generator_names]
        generator_names.extend(inversed_names)
        inversed = [inverse_permutation(p) for p in generators]
        generators.extend(inversed)
        central_state = list(range(len(generators[0])))
        return CayleyGraphDef.create(generators, central_state=central_state, generator_names=generator_names)
    else:
        raise ValueError(f"Unknown generator set: {name}")
