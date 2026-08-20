import pytest
import torch

from .models import ModelConfig
from .tokenizer import GroupTokenizer
from ..cayley_graph_def import CayleyGraphDef
from ..graphs_lib import MatrixGroups, PermutationGroups
from ..puzzles import Puzzles

# Megaminx: 20 corners with 3 stickers each, then 30 edges with 2 stickers each.
MEGAMINX_GROUPS = [[3, 20], [2, 30]]


def test_tokenizes_known_layout():
    # Two pieces with 2 stickers each (positions 0..3), then three pieces with 1 sticker each (positions 4..6).
    tokenizer = GroupTokenizer([[2, 2], [1, 3]])
    assert tokenizer.state_size == 7
    assert tokenizer.n_tokens == 5
    assert tokenizer.n_token_types == 2
    assert tokenizer.vocab_size == 4  # Segment of 2-sticker pieces has 4 possible tokens, segment of 1-sticker - 3.
    assert torch.equal(tokenizer.token_type_ids, torch.tensor([0, 0, 1, 1, 1]))

    # Solved state: first piece is in the first slot, and so on. Token of a slot is the value of its first element,
    # counted from the beginning of the segment (so the second segment's tokens are 4,5,6 minus offset 4).
    assert torch.equal(tokenizer(torch.tensor([0, 1, 2, 3, 4, 5, 6])), torch.tensor([0, 2, 0, 1, 2]))

    # Pieces swapped: second piece is in the first slot and rotated, first piece is in the second slot.
    assert torch.equal(tokenizer(torch.tensor([3, 2, 0, 1, 6, 4, 5])), torch.tensor([3, 0, 2, 0, 1]))


def test_megaminx_sizes():
    tokenizer = GroupTokenizer(MEGAMINX_GROUPS)
    assert tokenizer.groups == MEGAMINX_GROUPS
    assert tokenizer.state_size == 120
    assert tokenizer.n_tokens == 50  # 20 corners + 30 edges.
    assert tokenizer.vocab_size == 60  # 20 corners * 3 orientations = 30 edges * 2 orientations.
    assert tokenizer.token_type_ids.tolist() == [0] * 20 + [1] * 30

    graph_def = Puzzles.megaminx()
    tokens = tokenizer(torch.tensor(graph_def.central_state))
    # In the central state, i-th slot holds i-th piece in the default orientation.
    assert torch.equal(tokens, torch.tensor([3 * i for i in range(20)] + [2 * i for i in range(30)]))


def test_tokenizes_batch_of_states():
    graph_def = Puzzles.megaminx()
    tokenizer = GroupTokenizer(MEGAMINX_GROUPS)
    states = torch.tensor([_apply(graph_def, i) for i in range(graph_def.n_generators)])
    tokens = tokenizer(states)
    assert tokens.shape == (graph_def.n_generators, 50)
    assert tokens.dtype == torch.int64
    for i in range(graph_def.n_generators):
        assert torch.equal(tokens[i], tokenizer(states[i]))


def test_tokens_identify_states():
    # Tokenization is lossless for Megaminx: distinct states have distinct tokens.
    graph_def = Puzzles.megaminx()
    tokenizer = GroupTokenizer(MEGAMINX_GROUPS)
    states = torch.tensor(_walk(graph_def, 40))
    tokens = tokenizer(states)
    assert len(torch.unique(tokens, dim=0)) == len(torch.unique(states, dim=0))


def test_verify_accepts_puzzles_with_matching_grouping():
    GroupTokenizer(MEGAMINX_GROUPS).verify(Puzzles.megaminx())
    # Mini pyramorphix: 8 pieces with 3 stickers each.
    GroupTokenizer([[3, 8]]).verify(Puzzles.mini_pyramorphix())


def test_from_config():
    config = ModelConfig(
        model_type="MLP",
        input_size=120,
        num_classes_for_one_hot=120,
        layers_sizes=[64],
        tokenizer_groups=MEGAMINX_GROUPS,
    )
    tokenizer = GroupTokenizer.from_config(config)
    assert tokenizer.groups == MEGAMINX_GROUPS
    assert tokenizer.n_tokens == 50


def test_rejects_states_of_wrong_size():
    tokenizer = GroupTokenizer(MEGAMINX_GROUPS)
    with pytest.raises(ValueError, match="120 elements, got states with 119 elements"):
        tokenizer(torch.zeros((2, 119), dtype=torch.int64))
    with pytest.raises(ValueError, match="at least one dimension"):
        tokenizer(torch.tensor(0))


def test_rejects_inconsistent_groups():
    with pytest.raises(ValueError, match="must not be empty"):
        GroupTokenizer([])
    with pytest.raises(ValueError, match=r"pair \[group_size, num_groups\]"):
        GroupTokenizer([[3, 20, 1]])
    with pytest.raises(ValueError, match="must be positive"):
        GroupTokenizer([[3, 0]])
    with pytest.raises(ValueError, match="must be positive"):
        GroupTokenizer([[-3, 20]])


def test_from_config_rejects_config_without_tokenizer_groups():
    config = ModelConfig(model_type="MLP", input_size=120, num_classes_for_one_hot=120, layers_sizes=[64])
    with pytest.raises(ValueError, match="has no tokenizer_groups"):
        GroupTokenizer.from_config(config)


def test_from_config_rejects_groups_inconsistent_with_input_size():
    config = ModelConfig(
        model_type="MLP",
        input_size=100,
        num_classes_for_one_hot=120,
        layers_sizes=[64],
        tokenizer_groups=MEGAMINX_GROUPS,
    )
    with pytest.raises(ValueError, match="120 elements, but input_size in the config is 100"):
        GroupTokenizer.from_config(config)


def test_verify_rejects_wrong_grouping():
    # In this encoding of the 2x2x2 cube, stickers of one corner are not in consecutive positions.
    with pytest.raises(ValueError, match="elements of one piece, in cyclic order"):
        GroupTokenizer([[3, 8]]).verify(Puzzles.rubik_cube(2, "QTM"))

    # This generator swaps a piece of the first kind with a piece of the second kind.
    mixing_kinds = CayleyGraphDef.create([[2, 3, 0, 1]], central_state=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="outside of range"):
        GroupTokenizer([[2, 1], [2, 1]]).verify(mixing_kinds)

    # Generators of LRX shift elements one by one, so they do not keep pairs of elements together.
    with pytest.raises(ValueError, match='generator "L".* elements of one piece, in cyclic order'):
        GroupTokenizer([[2, 3]]).verify(PermutationGroups.lrx(6))

    # Elements of this central state are colors, not ids of stickers of pieces.
    with pytest.raises(ValueError, match="central state.* elements of one piece, in cyclic order"):
        GroupTokenizer([[2, 2]]).verify(PermutationGroups.lrx(4).with_central_state("0011"))


def test_verify_rejects_incompatible_graphs():
    with pytest.raises(ValueError, match="states of this graph have 120 elements"):
        GroupTokenizer([[3, 8]]).verify(Puzzles.megaminx())
    with pytest.raises(ValueError, match="permutation generators"):
        GroupTokenizer([[3, 3]]).verify(MatrixGroups.heisenberg())


def _apply(graph_def: CayleyGraphDef, generator_id: int, state=None) -> list[int]:
    """Applies generator with the given id to the given state (central state by default)."""
    if state is None:
        state = list(graph_def.central_state)
    return [state[i] for i in graph_def.generators_permutations[generator_id]]


def _walk(graph_def: CayleyGraphDef, num_steps: int) -> list[list[int]]:
    """Applies generators to the central state one by one, returning all visited states."""
    state = list(graph_def.central_state)
    states = [state]
    for step in range(num_steps):
        state = _apply(graph_def, (step * 7) % graph_def.n_generators, state)
        states.append(state)
    return states
