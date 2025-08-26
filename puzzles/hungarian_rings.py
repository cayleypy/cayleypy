from typing import List, Tuple


def _circular_shift(items: List, step: int) -> List:
    """With a positive step shifts list to the left"""
    if len(items) > 0:
        step = step % len(items)
    return items[step:] + items[:step]


def _get_intersections(left_index: int, right_index: int) -> int:
    if left_index == 0 and right_index == 0:
        intersections = 1
    elif left_index > 0 and right_index > 0:
        intersections = 2
    else:
        raise ValueError(f"Incorrect index value. left_index:{left_index} right_index:{right_index}")
    return intersections


def _create_right_ring(left_size: int, left_index: int, right_size: int, right_index: int, full_size: int) -> List:
    """Return indexes for the right ring only"""
    intersections = _get_intersections(left_index=left_index, right_index=right_index)
    if intersections == 2:
        second_intersection_index = left_size + right_size - right_index - 1
        right_ring = [0] + list(range(left_size, second_intersection_index)) + [left_index]
        if right_index > 1:
            right_ring = right_ring + list(range(second_intersection_index, full_size))
    else:
        right_ring = [0] + list(range(left_size, full_size))

    assert len(right_ring) == right_size
    return right_ring


def hungarian_rings_permutations(
    left_size: int, left_index: int, right_size: int, right_index: int, step: int = 1
) -> Tuple[List[int], List[int]]:
    """
    Creates permutations for left and right ring rotation. Rotation is clockwise with a positive step.
    Args:
        left_size: number of elements in the left ring
        left_index: Index of the second intersection on the left ring.
            Taken counterclockwise from the first intersection. If there is only one intersection, the value is zero
        right_size: number of elements in the right ring
        right_index: Index of the second intersection on the right ring. Taken clockwise from the first intersection.
        step: Rotation value in a clockwise direction.
            If the value is negative, the rotation is in the opposite direction.
    Returns:
        left_rotation: permutations for left ring rotation
        right_rotation: permutations for right ring rotation
    Example:
        >>> hungarian_rings_permutations(5, 2, 5, 2)
        [1, 2, 3, 4, 0, 5, 6, 7], [5, 1, 7, 3, 4, 6, 2, 0]
    """
    if left_index < 0 or right_index < 0:
        raise ValueError(f"Incorrect index value. left_index:{left_index} right_index:{right_index}")

    if left_size <= left_index or right_size <= right_index:
        raise ValueError(
            f"Ring size is too small. left_size:{left_size} left_index:{left_index} "
            + f"right_size:{right_size} right_index:{right_index}"
        )

    intersections = _get_intersections(left_index=left_index, right_index=right_index)
    full_size = left_size + right_size - intersections
    left_ring = list(range(0, left_size))
    left_rotation = _circular_shift(left_ring, step) + list(range(left_size, full_size))

    right_ring = _create_right_ring(left_size, left_index, right_size, right_index, full_size)

    shifted_right_ring = _circular_shift(right_ring, step)
    first_intersect_value = shifted_right_ring[0]
    shifted_right_ring.remove(first_intersect_value)
    second_intersect_value = None
    if intersections == 2:
        second_intersect_value = shifted_right_ring[-right_index]
        shifted_right_ring.remove(second_intersect_value)

    right_rotation = left_ring + shifted_right_ring
    right_rotation[0] = first_intersect_value
    if second_intersect_value is not None:
        right_rotation[left_index] = second_intersect_value

    return left_rotation, right_rotation


def get_santa_parameters_from_n(n: int) -> tuple[int, int, int, int]:
    """
    Parameters are similar to those used in the santa_2023 competition.
    The rings are the same size and intersect at one third(one fourth in santa). Indexes are shifted by 1.
    Returns:
        left_size: left ring size
        left_index: index of the second intersection on the left ring
        right_size: right ring size
        right_index: index of the second intersection on the right ring
    """
    right_size = (n + 2) // 2
    assert right_size >= 4
    left_size = (n + 2) - right_size
    left_index = right_size // 3  # the rings intersect with one third
    right_index = left_index + 1
    return left_size, left_index, right_size, right_index


def get_group(n: int):
    """
    Returns all non-repeating parameter variants for current n
    """
    result = []
    full_size_one_intersection = n + 1
    for left_size in range(2, full_size_one_intersection // 2 + 1):
        right_size = full_size_one_intersection - left_size
        result.append((left_size, 0, right_size, 0))
    full_size_two_intersections = n + 2
    for left_size in range(2, full_size_two_intersections // 2 + 1):
        right_size = full_size_two_intersections - left_size
        result.extend(get_pair_variants(left_size, right_size))
    return result


def get_pair_variants(left_size, right_size):
    """
    non-repeating index variants for two intersections
    """
    result = []
    for left_index in range(1, left_size // 2 + 1):
        for right_index in range(1, right_size // 2 + 1):
            parameters = (left_size, left_index, right_size, right_index)
            clone = (right_size, right_index, left_size, left_index)
            if parameters not in result and clone not in result:
                result.append(parameters)
    return result


def hungarian_rings_generators(
    left_size: int, left_index: int, right_size: int, right_index: int
) -> tuple[list[list[int]], list[str]]:
    """
    Generators for undirected graph. Rotate rings in both directions by 1 step.
    """
    if left_size <= 1 or right_size <= 1:
        raise ValueError(f"Ring size must be greater than 1. left_size:{left_size} right_size:{right_size}")

    forth_l, forth_r = hungarian_rings_permutations(
        left_size=left_size, left_index=left_index, right_size=right_size, right_index=right_index
    )
    back_l, back_r = hungarian_rings_permutations(
        left_size=left_size, left_index=left_index, right_size=right_size, right_index=right_index, step=-1
    )
    generators = [forth_l, forth_r]
    generator_names = ["L", "R"]
    if forth_l != back_l:
        generators.append(back_l)
        generator_names.append("-L")
    if forth_r != back_r:
        generators.append(back_r)
        generator_names.append("-R")

    return generators, generator_names
