"""Transformer model consuming states as sequences of tokens."""

import typing

import torch
from torch import nn

from .tokenizer import GroupTokenizer

if typing.TYPE_CHECKING:
    from .models import ModelConfig

# Number of features per attention head, used when the number of heads is not set explicitly in the config.
DEFAULT_FEATURES_PER_HEAD = 64

# Ratio of the feed-forward layer width to the model width, used when the former is not set explicitly in the config.
DEFAULT_FEEDFORWARD_MULTIPLIER = 4


class TransformerModel(nn.Module):
    """Transformer encoder over tokens of a state.

    A state is converted to tokens by :class:`cayleypy.models.GroupTokenizer` (one token per puzzle piece, as described
    by the `tokenizer_groups` field of the config). Tokens are embedded, their embeddings are summed with a learnable
    positional embedding and with an embedding of the token type (so the model can tell corners from edges), and the
    resulting sequence is processed by ``torch.nn.TransformerEncoder``. Encoded tokens are averaged, and a linear layer
    with `n_outputs` neurons is applied to the average.

    Blocks of the encoder are pre-norm (normalization before attention and before the feed-forward layer), use GELU
    activation and no dropout, so the model is deterministic in evaluation mode.

    The config describes this model as follows:

    - `layers_sizes` has one entry per encoder layer, and all entries must be equal (a transformer encoder keeps the
      same number of features in all layers), so ``[256] * 4`` means 4 layers of width 256;
    - `n_heads` and `dim_feedforward` are optional; by default there is one attention head per 64 features, and the
      feed-forward layer is 4 times wider than the model;
    - `tokenizer_groups` is required, and `num_classes_for_one_hot` is not used (the size of the vocabulary is
      determined by `tokenizer_groups`);
    - `n_outputs` set to the number of generators of a graph gives a Q-model, which estimates distances for all
      children of a state in a single forward pass (see :meth:`cayleypy.Predictor.score_children`).

    Output has shape ``[n_states]`` when ``n_outputs == 1``, and shape ``[n_states, n_outputs]`` otherwise.

    Example (Q-model for Megaminx, which has 120 elements in a state and 24 generators):
      >>> from cayleypy.models import ModelConfig
      >>> config = ModelConfig(
      ...     model_type="TRANSFORMER",
      ...     input_size=120,
      ...     num_classes_for_one_hot=60,
      ...     layers_sizes=[256] * 4,
      ...     n_outputs=24,
      ...     tokenizer_groups=[[3, 20], [2, 30]],
      ...     n_heads=8,
      ...     dim_feedforward=1024,
      ... )
      >>> model = config.build_model()
      >>> model.n_outputs
      24
    """

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        assert config.model_type == "TRANSFORMER"
        if config.n_outputs < 1:
            raise ValueError(f"n_outputs must be positive, got {config.n_outputs}.")
        if len(config.layers_sizes) == 0:
            raise ValueError("TransformerModel needs at least one encoder layer, but layers_sizes is empty.")
        if len(set(config.layers_sizes)) != 1:
            raise ValueError(
                f"All encoder layers must have the same width, got layers_sizes={config.layers_sizes}. Use "
                "[width] * num_layers."
            )
        d_model = config.layers_sizes[0]
        n_heads = config.n_heads if config.n_heads is not None else max(1, d_model // DEFAULT_FEATURES_PER_HEAD)
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if d_model % n_heads != 0:
            raise ValueError(f"Model width {d_model} must be divisible by the number of attention heads {n_heads}.")
        if config.dim_feedforward is not None:
            dim_feedforward = config.dim_feedforward
            if dim_feedforward < 1:
                raise ValueError(f"dim_feedforward must be positive, got {dim_feedforward}.")
        else:
            dim_feedforward = DEFAULT_FEEDFORWARD_MULTIPLIER * d_model

        self.n_outputs = config.n_outputs
        self.tokenizer = GroupTokenizer.from_config(config)
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.token_type_embedding = nn.Embedding(self.tokenizer.n_token_types, d_model)
        self.position_embedding = nn.Parameter(torch.empty(self.tokenizer.n_tokens, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=len(config.layers_sizes),
            norm=nn.LayerNorm(d_model),
            # All states have the same number of tokens and there is no padding, so nested tensors give nothing here.
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, self.n_outputs)

        # Not persistent: this is derived from `tokenizer_groups` in the config, so it must not be in the state dict.
        self.register_buffer("token_type_ids", self.tokenizer.token_type_ids, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_state = x.dim() == 1
        if single_state:
            x = x.unsqueeze(0)
        tokens = self.tokenizer(x)
        embedded = self.token_embedding(tokens) + self.token_type_embedding(self.token_type_ids)
        encoded = self.encoder(embedded + self.position_embedding)
        ans = self.head(encoded.mean(dim=1))
        # For single-output models the trailing dimension of size 1 is removed, so there is one score per state.
        if self.n_outputs == 1:
            ans = ans.squeeze(-1)
        return ans.squeeze(0) if single_state else ans
