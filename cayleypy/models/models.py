# pylint: disable=not-callable

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import kagglehub
import torch
from kagglehub import exceptions as kagglehub_exceptions
from torch import nn

# Errors kagglehub raises when weights cannot be downloaded: no network, no credentials, no such model. They are
# wrapped, because on their own they do not say which model of CayleyPy failed to load.
_KAGGLEHUB_ERRORS = (
    # Errors of the "requests" library, including kagglehub.exceptions.KaggleApiHTTPError, are subclasses of OSError.
    OSError,
    kagglehub_exceptions.BackendError,
    kagglehub_exceptions.CredentialError,
    kagglehub_exceptions.DataCorruptionError,
    kagglehub_exceptions.NotFoundError,
    kagglehub_exceptions.UnauthenticatedError,
)


def _download_from_kaggle(kaggle_id: str) -> str:
    """Downloads Kaggle model with weights, reporting failures as errors naming that model.

    :param kaggle_id: Id of the Kaggle model, as passed to `kagglehub.model_download`.
    :return: Path to the directory the model was downloaded to.
    """
    try:
        return kagglehub.model_download(kaggle_id)
    except _KAGGLEHUB_ERRORS as error:
        raise RuntimeError(
            f'Could not download weights from Kaggle model "{kaggle_id}": {error}. Downloading weights needs network '
            "access, and weights of a model that is not public also need Kaggle credentials to be configured (see "
            "https://github.com/Kaggle/kagglehub)."
        ) from error


def _load_state_dict(path: str, device: str) -> dict[str, Any]:
    """Loads state dict from a file with weights.

    Both bare state dicts and self-describing checkpoints written by :func:`cayleypy.models.save_checkpoint` are
    accepted, so weights saved in either format can be used for a model of :data:`PREDICTOR_MODELS`.

    :param path: Path to the file with weights.
    :param device: PyTorch device to load the weights to.
    :return: The state dict.
    """
    # `weights_only=True` is passed explicitly: files with weights contain only tensors and primitive values, so we
    # never need to unpickle arbitrary objects from them (and must not, because they are downloaded from the internet).
    data = torch.load(path, map_location=device, weights_only=True)
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


@dataclass(frozen=True)
class ModelConfig:
    """Configuration used to describe ML model.

    Fields `n_outputs`, `tokenizer_groups` and `graph_hash` describe capabilities added after the first version of this
    class. Their defaults describe a single-output model without tokenization, which is not tied to a particular graph,
    so configs written before these fields existed keep working.

    :param model_type: Type of the model, one of "MLP" (see :class:`MlpModel`) or "RESMLP"
        (see :class:`ResMlpModel`).
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
        else:
            raise ValueError("Unknown model type: " + self.model_type)

    def load(self, device="cpu") -> nn.Module:
        """Creates model described by this config and loads weights."""
        model = self.build_model()
        if self.weights_path is not None:
            path = self.weights_path
            if self.weights_kaggle_id is not None:
                path = os.path.join(_download_from_kaggle(self.weights_kaggle_id), path)
            model.load_state_dict(_load_state_dict(path, device))
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
