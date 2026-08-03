"""Model with two heads: one estimating distances for children of a state, another one for the state itself."""

import typing
from dataclasses import replace

import torch
from torch import nn

if typing.TYPE_CHECKING:
    from .models import ModelConfig


class QVModel(nn.Module):
    """Model with a Q-head (one output per generator) and a V-head (one output for the state itself).

    This is the architecture used by AlphaZero-like solvers: `Q(s)[i]` estimates the distance from the central state to
    the i-th child of `s`, and `V(s)` estimates the distance to `s` itself. Both heads are computed by the output layer
    of the backbone (its first `n_outputs` values are Q, the last one is V), which is equivalent to two separate linear
    heads over the features of the backbone and needs no parameters beyond them.

    The backbone can be any other architecture known to :meth:`cayleypy.models.ModelConfig.build_model`; its type is
    given by the `backbone_type` field of the config, and all other fields of the config describe it. So a config for
    this model is the config of its backbone, with `model_type` set to "QV", `backbone_type` set to the type of the
    backbone, and `n_outputs` set to the number of generators of the graph.

    Having both heads allows to check them against each other at inference time (v-consistency): a child lying on an
    optimal path from `s` must be at distance ``V(s)-1``, so a child whose Q-value disagrees with ``V(s)-1`` is
    inconsistent, and probably mispredicted. :meth:`score_children` adds penalty
    ``v_consistency_weight * |Q(s)[i] - (V(s)-1)|`` to the score of every child, which makes beam search prefer
    children that both heads agree about. The penalty is not used during training: it only reranks children, and with
    ``v_consistency_weight=0`` (the default) scores are the Q-values as is.

    :meth:`forward` returns the Q-values, so this model is used through :meth:`cayleypy.Predictor.score_children` (which
    calls :meth:`score_children` of this model, so the penalty is applied). Estimates for the states themselves are
    available through :meth:`v`.

    Example (Q+V model for a graph with 3 generators and 5 elements in a state):
      >>> from cayleypy.models import ModelConfig
      >>> config = ModelConfig(
      ...     model_type="QV",
      ...     input_size=5,
      ...     num_classes_for_one_hot=5,
      ...     layers_sizes=[64, 64],
      ...     n_outputs=3,
      ...     backbone_type="RESMLP",
      ...     v_consistency_weight=0.1,
      ... )
      >>> model = config.build_model()
      >>> model.n_outputs
      3
    """

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        assert config.model_type == "QV"
        if config.n_outputs < 1:
            raise ValueError(f"n_outputs must be positive, got {config.n_outputs}.")
        if config.backbone_type is None:
            raise ValueError(
                'QVModel needs a backbone, but backbone_type is not set in the config. Set it to e.g. "RESMLP".'
            )
        if config.backbone_type == "QV":
            raise ValueError('QVModel cannot be its own backbone, so backbone_type must not be "QV".')
        if config.v_consistency_weight < 0:
            raise ValueError(f"v_consistency_weight must be non-negative, got {config.v_consistency_weight}.")
        self.n_outputs = config.n_outputs
        self.v_consistency_weight = float(config.v_consistency_weight)
        # The backbone has one output more than this model: the extra output is the V-head.
        self.backbone = replace(config, model_type=config.backbone_type, n_outputs=self.n_outputs + 1).build_model()

    def heads(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies this model to `states`, returning outputs of both heads.

        :param states: States (in decoded representation) to apply this model to.
        :return: Pair (Q-values of shape ``[n_states, n_outputs]``, V-values of shape ``[n_states]``).
        """
        ans = self.backbone(states)
        return ans[..., :-1], ans[..., -1]

    def q(self, states: torch.Tensor) -> torch.Tensor:
        """Estimated distances for all children of `states`, of shape ``[n_states, n_outputs]``."""
        return self.heads(states)[0]

    def v(self, states: torch.Tensor) -> torch.Tensor:
        """Estimated distances for `states` themselves, of shape ``[n_states]``."""
        return self.heads(states)[1]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q(states)

    def score_children(self, states: torch.Tensor) -> torch.Tensor:
        """Estimates distances for all children of `states`, penalizing children inconsistent with the V-head.

        :param states: States (in decoded representation) whose children to score.
        :return: Tensor of shape ``[n_states, n_outputs]`` with estimated distances for children.
        """
        q, v = self.heads(states)
        if self.v_consistency_weight == 0:
            return q
        # A child of a state at distance V is at distance V-1 if it lies on an optimal path, so the further the Q-value
        # of a child is from V-1, the less the heads agree about that child.
        return q + self.v_consistency_weight * (q - (v.unsqueeze(-1) - 1.0)).abs()
