# pylint: disable=not-callable

import os
from dataclasses import dataclass
from typing import Any, Optional

import kagglehub
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration used to describe ML model."""

    model_type: str
    input_size: int
    num_classes_for_one_hot: int
    layers_sizes: list[int]
    weights_kaggle_id: Optional[str] = None
    weights_path: Optional[str] = None

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
        )

    def _build_model(self) -> nn.Module:
        if self.model_type == "MLP":
            return MlpModel(self)
        else:
            raise ValueError("Unknown model type: " + self.model_type)

    def load(self, device="cpu") -> nn.Module:
        """Creates model described by this config and loads weights."""
        model = self._build_model()
        if self.weights_path is not None:
            path = self.weights_path
            if self.weights_kaggle_id is not None:
                model_dir = kagglehub.model_download(self.weights_kaggle_id)
                path = os.path.join(model_dir, path)
            model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device)


class MlpModel(nn.Module):
    """Multi-layer perceptron model."""

    def __init__(self, config):
        super().__init__()
        assert config.model_type == "MLP"
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
