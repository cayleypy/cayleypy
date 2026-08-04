# pylint: disable=not-callable

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import kagglehub
import torch
from torch import nn

from .qv_model import QVModel


@dataclass(frozen=True)
class ModelConfig:
    """Configuration used to describe ML model.

    Fields `n_outputs`, `tokenizer_groups`, `graph_hash`, `backbone_type` and `v_consistency_weight` describe
    capabilities added after the first version of this class. Their defaults describe a single-output model without
    tokenization, which is not tied to a particular graph, so configs written before these fields existed keep working.

    :param model_type: Type of the model, one of "MLP" (see :class:`MlpModel`), "RESMLP"
        (see :class:`ResMlpModel`) or "QV" (see :class:`QVModel`).
    :param input_size: Number of elements in one state.
    :param num_classes_for_one_hot: Number of distinct values one element of a state can take.
    :param layers_sizes: Sizes of hidden layers (for "RESMLP" these are sizes of residual blocks).
    :param weights_kaggle_id: Id of the Kaggle model with weights (optional).
    :param weights_path: Path to the file with weights (optional).
    :param n_outputs: Number of outputs of the model. 1 means the model predicts distance for the state it is applied
        to. `n_generators` means the model predicts distance for every child of that state (Q-model).
    :param tokenizer_groups: Specification of how elements of a state are grouped into tokens, as a list of
        ``[group_size, num_groups]`` pairs. For example, ``[[3, 20], [2, 30]]`` means 20 tokens of 3 elements followed
        by 30 tokens of 2 elements. None means the state is not tokenized.
    :param graph_hash: Hash of the graph this model was trained for, see :func:`cayleypy.models.graph_hash`.
    :param backbone_type: Type of the backbone for models built on top of another architecture (only "QV" needs it).
        All other fields of this config describe that backbone.
    :param v_consistency_weight: Weight of the v-consistency penalty applied by :class:`QVModel` when it scores
        children. 0 means no penalty, and it must stay 0 unless the V-head of the model was supervised during training
        (see :class:`QVModel`).
    """

    model_type: str
    input_size: int
    num_classes_for_one_hot: int
    layers_sizes: list[int]
    weights_kaggle_id: Optional[str] = None
    weights_path: Optional[str] = None
    n_outputs: int = 1
    tokenizer_groups: Optional[list[list[int]]] = None
    graph_hash: Optional[str] = None
    backbone_type: Optional[str] = None
    v_consistency_weight: float = 0.0

    @staticmethod
    def from_dict(cfg: dict[str, Any]):
        """Creates config from Python dict."""
        return ModelConfig(
            model_type=cfg["model_type"],
            input_size=cfg["input_size"],
            num_classes_for_one_hot=cfg["num_classes_for_one_hot"],
            layers_sizes=cfg["layers_sizes"],
            weights_kaggle_id=cfg.get("weights_kaggle_id", None),
            weights_path=cfg.get("weights_path", None),
            n_outputs=cfg.get("n_outputs", 1),
            tokenizer_groups=cfg.get("tokenizer_groups", None),
            graph_hash=cfg.get("graph_hash", None),
            backbone_type=cfg.get("backbone_type", None),
            v_consistency_weight=cfg.get("v_consistency_weight", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts this config to a Python dict containing only primitive values."""
        return asdict(self)

    def build_model(self) -> nn.Module:
        """Creates model described by this config, with randomly initialized weights."""
        if self.model_type == "MLP":
            return MlpModel(self)
        elif self.model_type == "RESMLP":
            return ResMlpModel(self)
        elif self.model_type == "QV":
            return QVModel(self)
        else:
            raise ValueError("Unknown model type: " + self.model_type)

    def load(self, device="cpu") -> nn.Module:
        """Creates model described by this config and loads weights.

        Weights are loaded from `weights_path`. A config with neither `weights_path` nor `weights_kaggle_id` describes
        an untrained model, and the returned model has randomly initialized weights.

        :param device: PyTorch device to load the model to.
        :return: The model.
        """
        if self.weights_path is None and self.weights_kaggle_id is not None:
            # A Kaggle model is a directory, so without weights_path there is no way to tell which file in it holds the
            # weights - and silently returning a randomly initialized model instead is much worse than failing here.
            raise ValueError(
                f'Config has weights_kaggle_id="{self.weights_kaggle_id}" but no weights_path, so it is not known '
                "which file of that Kaggle model holds the weights. Set weights_path to the name of that file."
            )
        model = self.build_model()
        if self.weights_path is not None:
            path = self.weights_path
            if self.weights_kaggle_id is not None:
                model_dir = kagglehub.model_download(self.weights_kaggle_id)
                path = os.path.join(model_dir, path)
            # Weights in this format are bare state dicts, so we never need to unpickle arbitrary objects from them.
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.to(device)


def _one_hot_encode(states: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encodes elements of states and flattens the result, so it can be fed to a fully connected layer."""
    return nn.functional.one_hot(states.long(), num_classes=num_classes).float().flatten(start_dim=-2)


def _validate_n_outputs(config: ModelConfig) -> int:
    if config.n_outputs < 1:
        raise ValueError(f"n_outputs must be positive, got {config.n_outputs}.")
    return config.n_outputs


class MlpModel(nn.Module):
    """Multi-layer perceptron model.

    Consumes one-hot encoded states, applies hidden layers described by `layers_sizes` (each of them being
    Linear+LayerNorm+ReLU), then a linear output layer with `n_outputs` neurons.

    Output has shape ``[n_states]`` when ``n_outputs == 1``, and shape ``[n_states, n_outputs]`` otherwise.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.model_type == "MLP"
        self.n_outputs = _validate_n_outputs(config)
        self.num_classes_for_one_hot = config.num_classes_for_one_hot
        self.input_layer_size = config.input_size * self.num_classes_for_one_hot

        layers: list[nn.Module] = []
        in_features = self.input_layer_size
        for hidden_dim in config.layers_sizes:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        # For n_outputs=1 this layer has the same shape as before multi-output models were supported, so this model
        # still loads weights trained back then (including pretrained models from PREDICTOR_MODELS).
        layers.append(nn.Linear(in_features, self.n_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ans = self.layers(_one_hot_encode(x, self.num_classes_for_one_hot))
        # For single-output models the trailing dimension of size 1 is removed, so there is one score per state.
        return ans.squeeze(-1) if self.n_outputs == 1 else ans


class _ResBlock(nn.Module):
    """Residual block: Linear+LayerNorm+ReLU, adding its input to its output when their shapes match."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()
        # Skip connection is possible only when input and output of this block have the same number of features.
        self.has_skip = in_features == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ans = self.activation(self.norm(self.linear(x)))
        return x + ans if self.has_skip else ans


class ResMlpModel(nn.Module):
    """Multi-layer perceptron with residual (skip) connections.

    Deep perceptrons with skip connections are easier to train than plain ones, so this model is usually a better
    backbone than :class:`MlpModel` when many hidden layers are needed.

    Consumes one-hot encoded states, applies residual blocks described by `layers_sizes` (i.e. there are
    ``len(layers_sizes)`` blocks and i-th block has ``layers_sizes[i]`` neurons), then a linear output layer with
    `n_outputs` neurons. Each block is Linear+LayerNorm+ReLU and adds its input to its output. Blocks that change the
    number of features (e.g. the first block, which consumes the one-hot encoded state) have no skip connection, so
    ``layers_sizes=[512, 512, 512]`` means one projection followed by two residual blocks.

    Output has shape ``[n_states]`` when ``n_outputs == 1``, and shape ``[n_states, n_outputs]`` otherwise. Setting
    `n_outputs` to the number of generators of a graph gives a Q-model, which estimates distances for all children of
    a state in a single forward pass (see :meth:`cayleypy.Predictor.score_children`).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.model_type == "RESMLP"
        if len(config.layers_sizes) == 0:
            raise ValueError("ResMlpModel needs at least one block, but layers_sizes is empty.")
        self.n_outputs = _validate_n_outputs(config)
        self.num_classes_for_one_hot = config.num_classes_for_one_hot
        self.input_layer_size = config.input_size * self.num_classes_for_one_hot

        blocks: list[nn.Module] = []
        in_features = self.input_layer_size
        for hidden_dim in config.layers_sizes:
            blocks.append(_ResBlock(in_features, hidden_dim))
            in_features = hidden_dim
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(in_features, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ans = self.head(self.blocks(_one_hot_encode(x, self.num_classes_for_one_hot)))
        # For single-output models the trailing dimension of size 1 is removed, so there is one score per state.
        return ans.squeeze(-1) if self.n_outputs == 1 else ans
