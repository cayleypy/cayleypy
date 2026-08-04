# pylint: disable=not-callable

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import kagglehub
import torch
from torch import nn

from .transformer import TransformerModel


@dataclass(frozen=True)
class ModelConfig:
    """Configuration used to describe ML model.

    Fields `n_outputs`, `tokenizer_groups` and `graph_hash` describe capabilities added after the first version of this
    class. Their defaults describe a single-output model without tokenization, which is not tied to a particular graph,
    so configs written before these fields existed keep working.

    :param model_type: Type of the model, e.g. "MLP".
    :param input_size: Number of elements in one state.
    :param num_classes_for_one_hot: Number of distinct values one element of a state can take.
    :param layers_sizes: Sizes of hidden layers.
    :param weights_kaggle_id: Id of the Kaggle model with weights (optional).
    :param weights_path: Path to the file with weights (optional).
    :param n_outputs: Number of outputs of the model. 1 means the model predicts distance for the state it is applied
        to. `n_generators` means the model predicts distance for every child of that state (Q-model).
    :param tokenizer_groups: Specification of how elements of a state are grouped into tokens, as a list of
        ``[group_size, num_groups]`` pairs. For example, ``[[3, 20], [2, 30]]`` means 20 tokens of 3 elements followed
        by 30 tokens of 2 elements. None means the state is not tokenized.
    :param graph_hash: Hash of the graph this model was trained for, see :func:`cayleypy.models.graph_hash`.
    :param n_heads: Number of attention heads, for models with attention. None means one head per 64 features.
    :param dim_feedforward: Width of the feed-forward layer inside a transformer block. None means 4 times the width
        of the model.
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
    n_heads: Optional[int] = None
    dim_feedforward: Optional[int] = None

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
            n_heads=cfg.get("n_heads", None),
            dim_feedforward=cfg.get("dim_feedforward", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts this config to a Python dict containing only primitive values."""
        return asdict(self)

    def build_model(self) -> nn.Module:
        """Creates model described by this config, with randomly initialized weights."""
        if self.model_type == "MLP":
            return MlpModel(self)
        elif self.model_type == "TRANSFORMER":
            return TransformerModel(self)
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


class MlpModel(nn.Module):
    """Multi-layer perceptron model."""

    def __init__(self, config):
        super().__init__()
        assert config.model_type == "MLP"
        if config.n_outputs != 1:
            raise ValueError(f"MlpModel supports only n_outputs=1, got {config.n_outputs}.")
        self.num_classes_for_one_hot = config.num_classes_for_one_hot
        self.input_layer_size = config.input_size * self.num_classes_for_one_hot

        layers = []
        in_features = self.input_layer_size
        for hidden_dim in config.layers_sizes:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.one_hot(x.long(), num_classes=self.num_classes_for_one_hot).float().flatten(start_dim=-2)
        return self.layers(x).squeeze(-1)
