import os

import pytest
import torch

from .cayley_graph import CayleyGraph
from .graphs_lib import PermutationGroups, MatrixGroups
from .predictor import Predictor
from .puzzles import Puzzles
from .symmetries import SymmetryGroup, SymmetrizedPredictor

RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"


class AsymmetricModel(torch.nn.Module):
    """Model whose predictions are deliberately not symmetric (they depend on positions of elements)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * torch.arange(1, x.shape[1] + 1)).sum(dim=1).float()


def exact_distances(graph: CayleyGraph) -> dict[tuple[int, ...], int]:
    """Returns exact distances from the central state to all states, computed by BFS."""
    bfs_result = graph.bfs(max_layer_size_to_store=None)
    ans = {}
    for layer_id, layer in bfs_result.layers.items():
        for state in layer:
            ans[tuple(int(x) for x in state)] = layer_id
    return ans


def test_reflections_for_lrx():
    graph_def = PermutationGroups.lrx(5)
    symmetry_group = SymmetryGroup.reflections(graph_def)
    assert symmetry_group.n_symmetries == 2
    assert symmetry_group.symmetries == [[0, 1, 2, 3, 4], [1, 0, 4, 3, 2]]

    # The reflection swaps L and R, and preserves X.
    assert symmetry_group.transport_actions(0) == [0, 1, 2]
    assert symmetry_group.transport_actions(1) == [1, 0, 2]

    # The central state is the identity permutation, so relabeling of elements is inverse of the symmetry.
    assert symmetry_group.element_maps == [[0, 1, 2, 3, 4], [1, 0, 4, 3, 2]]
    symmetry_group.verify()


def test_reflections_for_lrx_with_shifted_transposition():
    # Here X is transposition of elements 0 and 2, so the reflection is i -> 2-i (mod 5).
    graph_def = PermutationGroups.lrx(5, k=2)
    symmetry_group = SymmetryGroup.reflections(graph_def)
    assert symmetry_group.symmetries == [[0, 1, 2, 3, 4], [2, 1, 0, 4, 3]]
    assert symmetry_group.transport_actions(1) == [1, 0, 2]


def test_reflections_for_coset_graph():
    # The central state is symmetric with respect to the reflection, so elements are not relabeled.
    graph_def = PermutationGroups.lrx(5).with_central_state([0, 0, 1, 1, 1])
    symmetry_group = SymmetryGroup.reflections(graph_def)
    assert symmetry_group.symmetries == [[0, 1, 2, 3, 4], [1, 0, 4, 3, 2]]
    assert symmetry_group.element_maps[1] == [0, 1]


@pytest.mark.parametrize("metric", ["QTM", "QSTM", "HTM", "ATM"])
def test_rubik_cube_rotations(metric):
    graph_def = Puzzles.rubik_cube(2, metric=metric)
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    assert symmetry_group.n_symmetries == 24
    symmetry_group.verify()

    # Rotating the whole cube permutes its 6 faces, and all 6 faces must be moved by some rotation.
    element_maps = symmetry_group.element_maps
    assert element_maps[0] == list(range(6))
    for face in range(6):
        assert any(m[face] != face for m in element_maps)


def test_apply_preserves_central_state():
    lrx = PermutationGroups.lrx(6)
    cube = Puzzles.rubik_cube(2, metric="QTM")
    for graph_def, symmetry_group in [
        (lrx, SymmetryGroup.reflections(lrx)),
        (cube, SymmetryGroup.rubik_cube_rotations(cube)),
    ]:
        central_state = torch.tensor(graph_def.central_state)
        for i in range(symmetry_group.n_symmetries):
            assert torch.equal(symmetry_group.apply(central_state, i), central_state)


def test_apply_single_state_and_batch():
    graph_def = PermutationGroups.lrx(5)
    symmetry_group = SymmetryGroup.reflections(graph_def)
    state = torch.tensor([0, 2, 1, 3, 4])

    # The state is transposition of elements 1 and 2. Reflection i -> 1-i (mod 5) maps 1 to 0 and 2 to 4, so the image
    # must be transposition of elements 0 and 4.
    assert symmetry_group.apply(state, 1).tolist() == [4, 1, 2, 3, 0]
    assert symmetry_group.apply(state.reshape((1, 5)), 1).tolist() == [[4, 1, 2, 3, 0]]

    # Applying the same symmetry twice gives the original state back (it is an involution).
    assert torch.equal(symmetry_group.apply(symmetry_group.apply(state, 1), 1), state)


def test_transport_actions_matches_generators():
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    graph = CayleyGraph(graph_def, device="cpu", random_seed=0)
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    states = graph.random_walks(width=20, length=6)[0]

    for k in range(symmetry_group.n_symmetries):
        images = symmetry_group.apply(states, k)
        transport = symmetry_group.transport_actions(k)
        for i in range(graph_def.n_generators):
            # Image of a child obtained by applying generator i is a child of the image, obtained by generator j.
            expected = symmetry_group.apply(graph.apply_path(states, [i]), k)
            assert torch.equal(expected, graph.apply_path(images, [transport[i]]))


def test_symmetries_preserve_distances():
    graph_def = PermutationGroups.lrx(6)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    distances = exact_distances(graph)
    assert len(distances) == 720

    states = torch.tensor(list(distances.keys()))
    expected = torch.tensor(list(distances.values()))
    for i in range(symmetry_group.n_symmetries):
        images = symmetry_group.apply(states, i)
        actual = torch.tensor([distances[tuple(int(x) for x in state)] for state in images])
        assert torch.equal(actual, expected)


def test_transport_actions_against_exact_bfs():
    graph_def = PermutationGroups.lrx(6)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    distances = exact_distances(graph)
    states = torch.tensor(list(distances.keys()))

    for k in range(symmetry_group.n_symmetries):
        images = symmetry_group.apply(states, k)
        transport = symmetry_group.transport_actions(k)
        for i in range(graph_def.n_generators):
            children = graph.apply_path(states, [i])
            transported_children = graph.apply_path(images, [transport[i]])
            for child, transported_child in zip(children, transported_children):
                assert distances[tuple(int(x) for x in child)] == distances[tuple(int(x) for x in transported_child)]


def test_cube_rotations_map_bfs_layers_onto_themselves():
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    bfs_result = graph.bfs(max_layer_size_to_store=None, max_diameter=4)

    for layer_id, layer in bfs_result.layers.items():
        layer_hashes = set(graph.hasher.make_hashes(graph.encode_states(layer)).tolist())
        assert len(layer_hashes) == layer.shape[0]
        for i in range(symmetry_group.n_symmetries):
            images = symmetry_group.apply(layer, i)
            image_hashes = graph.hasher.make_hashes(graph.encode_states(images))
            # Symmetries preserve distances, so images of a BFS layer are exactly the same layer.
            assert set(image_hashes.tolist()) == layer_hashes, f"layer {layer_id}, symmetry {i}"


def test_tta_does_not_change_symmetric_predictor():
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    graph = CayleyGraph(graph_def, device="cpu", random_seed=0)
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    predictor = Predictor(graph, "hamming")
    symmetrized = SymmetrizedPredictor(predictor, symmetry_group)
    states = graph.random_walks(width=10, length=5)[0]

    # Hamming distance is invariant under symmetries, so averaging over symmetries changes nothing.
    assert torch.equal(symmetrized(states), predictor(states).float())
    assert torch.equal(symmetrized.score_children(states), predictor.score_children(states).float())


def test_tta_averages_over_symmetries():
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    predictor = Predictor(graph, AsymmetricModel())
    symmetrized = SymmetrizedPredictor(predictor, symmetry_group)
    states = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 0, 2, 4, 3]])

    expected = sum(predictor(symmetry_group.apply(states, i)) for i in range(2)) / 2
    assert torch.allclose(symmetrized(states), expected)

    expected_children = torch.stack(
        [
            symmetry_group.transport_scores(predictor.score_children(symmetry_group.apply(states, i)), i)
            for i in range(2)
        ]
    ).mean(dim=0)
    assert torch.allclose(symmetrized.score_children(states), expected_children)


def test_tta_predictions_are_symmetric():
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    graph = CayleyGraph(graph_def, device="cpu", random_seed=0)
    symmetry_group = SymmetryGroup.rubik_cube_rotations(graph_def)
    symmetrized = SymmetrizedPredictor(Predictor(graph, AsymmetricModel()), symmetry_group)
    states = graph.random_walks(width=10, length=5)[0]

    scores = symmetrized(states)
    child_scores = symmetrized.score_children(states)
    for k in range(symmetry_group.n_symmetries):
        images = symmetry_group.apply(states, k)
        # Predictions of a symmetrized predictor are the same for all states in one orbit.
        assert torch.allclose(symmetrized(images), scores, atol=1e-4)
        image_child_scores = symmetry_group.transport_scores(symmetrized.score_children(images), k)
        assert torch.allclose(image_child_scores, child_scores, atol=1e-4)


def test_tta_reduces_error_of_asymmetric_predictor():
    graph_def = PermutationGroups.lrx(6)
    graph = CayleyGraph(graph_def, device="cpu")
    symmetry_group = SymmetryGroup.reflections(graph_def)
    distances = exact_distances(graph)
    states = torch.tensor(list(distances.keys()))
    true_distances = torch.tensor([distances[tuple(int(x) for x in state)] for state in states]).float()

    # Predictor that knows exact distances, but adds error depending on positions of elements (i.e. not symmetric).
    def predict(x: torch.Tensor) -> torch.Tensor:
        exact = torch.tensor([distances[tuple(int(y) for y in state)] for state in x]).float()
        return exact + ((x[:, 0] * 7 + x[:, 1] * 13 + x[:, 2] * 3) % 7).float() - 3.0

    predictor = Predictor(graph, predict)
    symmetrized = SymmetrizedPredictor(predictor, symmetry_group)
    base_error = float((predictor(states) - true_distances).abs().mean())
    tta_error = float((symmetrized(states) - true_distances).abs().mean())

    # Errors for a state and for its image partially cancel each other, so averaging them reduces the error.
    assert base_error > 1.5
    assert tta_error < 0.8 * base_error


def test_symmetrized_predictor_in_beam_search():
    graph_def = PermutationGroups.lrx(6)
    graph = CayleyGraph(graph_def, device="cpu", random_seed=0)
    symmetry_group = SymmetryGroup.reflections(graph_def)
    symmetrized = SymmetrizedPredictor(Predictor(graph, "hamming"), symmetry_group)
    start_state = graph.random_walks(width=1, length=20)[0][-1]

    result = graph.beam_search(start_state=start_state, predictor=symmetrized, beam_width=1000, return_path=True)
    assert result.path_found
    graph.validate_path(start_state, result.path)


def test_derive_finds_all_symmetries():
    graph_def = PermutationGroups.lrx(5)
    symmetry_group = SymmetryGroup.derive(graph_def)
    assert symmetry_group.symmetries == [[0, 1, 2, 3, 4], [1, 0, 4, 3, 2]]
    symmetry_group.verify()

    # For a coset graph with a symmetric central state, the same permutations are symmetries.
    coset_graph_def = graph_def.with_central_state([0, 0, 1, 1, 1])
    assert SymmetryGroup.derive(coset_graph_def).symmetries == symmetry_group.symmetries


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="slow test")
def test_derive_for_larger_graph():
    graph_def = PermutationGroups.lrx(8)
    symmetry_group = SymmetryGroup.derive(graph_def)
    assert symmetry_group.symmetries == [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2]]
    symmetry_group.verify()

    # All symmetries of the 8-element cyclic graph are its 8 rotations and 8 reflections.
    symmetry_group = SymmetryGroup.derive(PermutationGroups.cyclic_coxeter(8))
    assert symmetry_group.n_symmetries == 16
    symmetry_group.verify()


def test_derive_rejects_large_state_size():
    with pytest.raises(ValueError, match="Brute force is too slow"):
        SymmetryGroup.derive(PermutationGroups.lrx(10))


def test_verify_catches_non_symmetry():
    graph_def = PermutationGroups.lrx(5)
    identity = [0, 1, 2, 3, 4]

    # This permutation does not turn generators into generators (its conjugate of L is not a generator).
    symmetry_group = SymmetryGroup([identity, [1, 2, 0, 3, 4]], graph_def)
    with pytest.raises(ValueError, match="conjugate of generator L"):
        symmetry_group.verify()

    # This permutation is a symmetry, but the set is not closed under composition.
    symmetry_group = SymmetryGroup([[1, 0, 4, 3, 2]], graph_def)
    with pytest.raises(ValueError, match="Identity permutation"):
        symmetry_group.verify()


def test_verify_catches_non_closed_group():
    graph_def = Puzzles.rubik_cube(2, metric="QTM")
    all_rotations = SymmetryGroup.rubik_cube_rotations(graph_def).symmetries

    # One rotation around an axis has order 4, so it does not form a group with the identity alone.
    symmetry_group = SymmetryGroup([all_rotations[0], all_rotations[1]], graph_def)
    with pytest.raises(ValueError, match="not closed under composition"):
        symmetry_group.verify()


def test_verify_catches_non_preserved_central_state():
    graph_def = PermutationGroups.lrx(5).with_central_state([0, 0, 1, 1, 1])

    # This permutation preserves generators, but maps an element of the central state to 2 different elements.
    symmetry_group = SymmetryGroup([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4]], graph_def)
    with pytest.raises(ValueError, match="does not preserve the central state"):
        symmetry_group.verify()


def test_reflections_when_there_is_no_reflection_symmetry():
    # LX generators are not inverse-closed, so no reflection turns generators into generators.
    with pytest.raises(ValueError, match="No reflection"):
        SymmetryGroup.reflections(PermutationGroups.lx(5))


def test_rubik_cube_rotations_for_wrong_graph():
    with pytest.raises(ValueError, match="hardcoded only for the 2x2x2 cube"):
        SymmetryGroup.rubik_cube_rotations(Puzzles.rubik_cube(3, metric="QTM"))

    # This graph has states of the right size, but the rotations are not its symmetries.
    with pytest.raises(ValueError, match="not a symmetry"):
        SymmetryGroup.rubik_cube_rotations(PermutationGroups.lrx(24))


def test_symmetry_group_rejects_invalid_input():
    graph_def = PermutationGroups.lrx(5)
    with pytest.raises(ValueError, match="At least one symmetry"):
        SymmetryGroup([], graph_def)
    with pytest.raises(ValueError, match="is not a permutation"):
        SymmetryGroup([[0, 1, 2, 3]], graph_def)
    with pytest.raises(ValueError, match="is not a permutation"):
        SymmetryGroup([[0, 1, 2, 3, 3]], graph_def)
    with pytest.raises(ValueError, match="must be distinct"):
        SymmetryGroup([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], graph_def)
    with pytest.raises(ValueError, match="only for graphs whose generators are permutations"):
        SymmetryGroup([[0, 1, 2, 3, 4]], MatrixGroups.heisenberg())


def test_apply_rejects_invalid_input():
    graph_def = PermutationGroups.lrx(5)
    symmetry_group = SymmetryGroup.reflections(graph_def)
    with pytest.raises(ValueError, match="symmetry_id must be between 0 and 1"):
        symmetry_group.apply(torch.tensor([0, 1, 2, 3, 4]), 2)
    with pytest.raises(ValueError, match="Expected states of size 5"):
        symmetry_group.apply(torch.tensor([0, 1, 2, 3]), 0)
    with pytest.raises(ValueError, match="Expected states of size 5"):
        symmetry_group.apply(torch.tensor(0), 0)
    with pytest.raises(ValueError, match="Expected states of size 5"):
        symmetry_group.apply(torch.zeros((2, 2, 5), dtype=torch.int64), 0)


def test_transport_scores_rejects_invalid_input():
    graph_def = PermutationGroups.lrx(5)
    symmetry_group = SymmetryGroup.reflections(graph_def)
    with pytest.raises(ValueError, match=r"Expected child scores of shape \[n_states, 3\]"):
        symmetry_group.transport_scores(torch.zeros((2, 4)), 0)
    with pytest.raises(ValueError, match=r"Expected child scores of shape \[n_states, 3\]"):
        symmetry_group.transport_scores(torch.zeros((2,)), 0)


def test_symmetrized_predictor_rejects_group_for_other_graph():
    graph = CayleyGraph(PermutationGroups.lrx(5), device="cpu")
    predictor = Predictor(graph, "hamming")

    # Same state size, but different generators.
    symmetry_group = SymmetryGroup.reflections(PermutationGroups.lrx(5, k=2))
    with pytest.raises(ValueError, match="different graph"):
        SymmetrizedPredictor(predictor, symmetry_group)

    symmetry_group = SymmetryGroup.reflections(PermutationGroups.lrx(5).with_central_state([0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="different graph"):
        SymmetrizedPredictor(predictor, symmetry_group)


def test_symmetrized_predictor_rejects_symmetries_that_are_not_a_group():
    """Test that TTA over a set that is not a group is rejected (its predictions would not be symmetric)."""
    graph_def = PermutationGroups.lrx(5)
    graph = CayleyGraph(graph_def, device="cpu")
    predictor = Predictor(graph, "hamming")

    # A permutation that is not a symmetry of this graph (its conjugate of L is not a generator).
    with pytest.raises(ValueError, match="conjugate of generator L"):
        SymmetrizedPredictor(predictor, SymmetryGroup([[0, 1, 2, 3, 4], [1, 2, 0, 3, 4]], graph_def))

    # A genuine symmetry, but the set does not contain the identity, so it is not a group.
    with pytest.raises(ValueError, match="Identity permutation"):
        SymmetrizedPredictor(predictor, SymmetryGroup([[1, 0, 4, 3, 2]], graph_def))


def test_symmetrized_q_model_is_a_q_model():
    """Test that TTA over a Q-model reports its outputs, so callers score children rather than states."""
    graph_def = PermutationGroups.lrx(6)
    graph = CayleyGraph(graph_def, device="cpu")
    n_generators = graph_def.n_generators

    class _QModel(torch.nn.Module):
        n_outputs = n_generators

        def forward(self, states: torch.Tensor) -> torch.Tensor:
            return states[:, :1].float() + torch.arange(n_generators, dtype=torch.float32)

    base = Predictor(graph, _QModel())
    symmetrized = SymmetrizedPredictor(base, SymmetryGroup.reflections(graph_def))

    assert symmetrized.n_outputs == n_generators
    assert symmetrized.score_children(torch.tensor([[1, 0, 2, 3, 4, 5]])).shape == (1, n_generators)
