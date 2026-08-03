"""Tokenization of states for models that process them as sequences of tokens."""

import typing
from typing import Sequence

import torch

from .models import ModelConfig

if typing.TYPE_CHECKING:
    from ..cayley_graph_def import CayleyGraphDef


class GroupTokenizer:
    """Converts states to sequences of tokens, one token per group of consecutive elements of a state.

    States of puzzles are usually encoded as permutations of sticker ids, where stickers of one piece occupy
    consecutive positions. Such state can be described more compactly by one token per piece: the token says which
    piece is in this slot and how it is oriented. For example, the Megaminx state has 120 elements: 20 corners with
    3 stickers each, followed by 30 edges with 2 stickers each. It is described by 50 tokens (one per piece), and each
    token takes one of 60 values (20 corners times 3 orientations, or 30 edges times 2 orientations). Such tokenization
    is described by groups ``[[3, 20], [2, 30]]``, see :attr:`cayleypy.models.ModelConfig.tokenizer_groups`.

    The i-th token is the value of the first element of the i-th group, counted from the beginning of the segment this
    group belongs to (a segment is all groups described by one ``[group_size, num_groups]`` pair). Therefore token ids
    of one segment are in range ``[0, group_size * num_groups)``, and token ids of different segments overlap. Models
    can tell segments apart by position, or by adding an embedding of `token_type_ids` to the embedding of tokens.

    This encodes the state without loss of information as long as elements of one group are always stickers of one
    piece, listed in the same cyclic order - then the whole group is determined by its first element. Use
    :meth:`verify` to check this for a particular graph.

    Example:
      >>> from cayleypy.models import GroupTokenizer
      >>> tokenizer = GroupTokenizer([[3, 20], [2, 30]])  # Megaminx.
      >>> tokenizer.n_tokens, tokenizer.vocab_size
      (50, 60)
    """

    def __init__(self, groups: Sequence[Sequence[int]]):
        """Creates tokenizer for states grouped as described by `groups`.

        :param groups: Specification of groups, as a list of ``[group_size, num_groups]`` pairs. Groups are laid out in
            the state consecutively, in the order they are listed here.
        """
        if len(groups) == 0:
            raise ValueError("Groups must not be empty.")
        parsed: list[list[int]] = []
        for spec in groups:
            if len(spec) != 2:
                raise ValueError(f"Group must be described by pair [group_size, num_groups], got {list(spec)}.")
            group_size, num_groups = int(spec[0]), int(spec[1])
            if group_size < 1 or num_groups < 1:
                raise ValueError(f"Group size and number of groups must be positive, got {[group_size, num_groups]}.")
            parsed.append([group_size, num_groups])

        self.groups: list[list[int]] = parsed
        """Groups this tokenizer was created with, as a list of ``[group_size, num_groups]`` pairs."""

        self.state_size: int = sum(group_size * num_groups for group_size, num_groups in parsed)
        """Number of elements in the state this tokenizer expects."""

        self.n_tokens: int = sum(num_groups for _, num_groups in parsed)
        """Number of tokens one state is converted to."""

        self.n_token_types: int = len(parsed)
        """Number of segments (i.e. of ``[group_size, num_groups]`` pairs describing this tokenization)."""

        self.vocab_size: int = max(group_size * num_groups for group_size, num_groups in parsed)
        """Number of distinct values a token can take (i.e. size of vocabulary needed to embed tokens)."""

        first_elements: list[int] = []
        value_offsets: list[int] = []
        token_types: list[int] = []
        position = 0
        for token_type, (group_size, num_groups) in enumerate(parsed):
            segment_start = position
            for _ in range(num_groups):
                first_elements.append(position)
                value_offsets.append(segment_start)
                token_types.append(token_type)
                position += group_size
        self._first_elements = torch.tensor(first_elements, dtype=torch.int64)
        self._value_offsets = torch.tensor(value_offsets, dtype=torch.int64)

        self.token_type_ids: torch.Tensor = torch.tensor(token_types, dtype=torch.int64)
        """Index of the segment each token belongs to, of shape ``[n_tokens]``."""

    @staticmethod
    def from_config(config: ModelConfig) -> "GroupTokenizer":
        """Creates tokenizer described by `tokenizer_groups` field of the given model config.

        :param config: Config of a model that consumes tokens.
        :return: Tokenizer described by this config.
        """
        if config.tokenizer_groups is None:
            raise ValueError(
                f'Config of model of type "{config.model_type}" has no tokenizer_groups, but this model needs '
                "tokenized states. Set tokenizer_groups to a list of [group_size, num_groups] pairs."
            )
        tokenizer = GroupTokenizer(config.tokenizer_groups)
        if tokenizer.state_size != config.input_size:
            raise ValueError(
                f"Groups {tokenizer.groups} describe states with {tokenizer.state_size} elements, but input_size in "
                f"the config is {config.input_size}."
            )
        return tokenizer

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Converts states to tokens.

        :param states: States in "decoded" format, of shape ``[n_states, state_size]`` or ``[state_size]``.
        :return: Tokens (int64), of shape ``[n_states, n_tokens]`` or ``[n_tokens]`` respectively.
        """
        if states.dim() == 0:
            raise ValueError("States must have at least one dimension.")
        if states.shape[-1] != self.state_size:
            raise ValueError(
                f"Groups {self.groups} describe states with {self.state_size} elements, got states with "
                f"{states.shape[-1]} elements."
            )
        return states[..., self._first_elements.to(states.device)].to(torch.int64) - self._value_offsets.to(
            states.device
        )

    def verify(self, graph_def: "CayleyGraphDef") -> None:
        """Checks that states of the given graph are tokenized without loss of information.

        This holds when the central state lists stickers of every piece in one group, and every generator moves
        stickers of one piece to positions of one piece, preserving their cyclic order. Then every state reachable from
        the central state has one piece per group, so the group is determined by its first element (i.e. by its token).

        :param graph_def: Definition of the graph whose states are going to be tokenized.
        :raises ValueError: If states of this graph cannot be tokenized this way.
        """
        if not graph_def.is_permutation_group():
            raise ValueError("Tokenization is supported only for graphs with permutation generators.")
        if graph_def.state_size != self.state_size:
            raise ValueError(
                f"Groups {self.groups} describe states with {self.state_size} elements, but states of this graph have "
                f"{graph_def.state_size} elements."
            )
        self._verify_pieces([int(x) for x in graph_def.central_state], "central state")
        for name, permutation in zip(graph_def.generator_names, graph_def.generators_permutations):
            self._verify_pieces([int(x) for x in permutation], f'generator "{name}"')

    def _verify_pieces(self, values: list[int], what: str) -> None:
        """Checks that `values` list elements of one piece in every group, in cyclic order."""
        position = 0
        for group_size, num_groups in self.groups:
            segment_start = position
            segment_end = position + group_size * num_groups
            for _ in range(num_groups):
                group = values[position : position + group_size]
                first = group[0]
                if not segment_start <= first < segment_end:
                    raise ValueError(
                        f"In {what}, element at position {position} is {first}, which is outside of range "
                        f"[{segment_start}, {segment_end}) of its segment (pieces of different kinds must not mix)."
                    )
                base = segment_start + (first - segment_start) // group_size * group_size
                rotation = (first - segment_start) % group_size
                expected = [base + (j + rotation) % group_size for j in range(group_size)]
                if group != expected:
                    raise ValueError(
                        f"In {what}, elements at positions {position}..{position + group_size - 1} are {group}, but "
                        f"expected {expected}: elements of one group must be elements of one piece, in cyclic order."
                    )
                position += group_size
