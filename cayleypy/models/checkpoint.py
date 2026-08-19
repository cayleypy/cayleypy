"""Self-describing checkpoints for models used as predictors."""

import hashlib
import json
import os
import typing
from dataclasses import replace
from typing import Any, Optional, Union

import torch
from torch import nn

from .models import ModelConfig

if typing.TYPE_CHECKING:
    from ..cayley_graph_def import CayleyGraphDef

# Version of the checkpoint format. It is incremented when the format changes in a non-backward-compatible way.
CHECKPOINT_FORMAT_VERSION = 1

PathType = Union[str, os.PathLike]


def graph_hash(graph_def: "CayleyGraphDef") -> str:
    """Computes hash of the mathematical definition of a graph.

    Only data affecting mathematical properties of the graph is hashed: type of generators, the generators themselves,
    and the central state. Names of the graph and of its generators are deliberately not hashed, so renaming a graph
    does not invalidate checkpoints of models trained for it.

    :param graph_def: Definition of the graph.
    :return: Hexadecimal sha256 hash of the definition.
    """
    generators: Any
    if graph_def.is_permutation_group():
        generators = [[int(x) for x in perm] for perm in graph_def.generators_permutations]
    else:
        generators = [[g.matrix.tolist(), int(g.modulo)] for g in graph_def.generators_matrices]
    data = {
        "generators_type": graph_def.generators_type.name,
        "generators": generators,
        "central_state": [int(x) for x in graph_def.central_state],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def save_checkpoint(
    path: PathType,
    model: nn.Module,
    config: ModelConfig,
    graph_def: Optional["CayleyGraphDef"] = None,
) -> ModelConfig:
    """Saves weights of a model together with the config describing this model.

    Unlike a bare state dict, such checkpoint is self-describing: :func:`load_checkpoint` recreates the model from it
    without knowing the architecture in advance.

    :param path: Path to the file to write.
    :param model: Model whose weights to save.
    :param config: Config describing `model`.
    :param graph_def: Definition of the graph this model was trained for (optional). If given, hash of this definition
        is stored in the checkpoint, and :func:`load_checkpoint` will check it.
    :return: Config stored in the checkpoint (it differs from `config` when `graph_def` is given).
    """
    if graph_def is not None:
        config = replace(config, graph_hash=graph_hash(graph_def))
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    return config


def load_checkpoint(
    path: PathType,
    device: str = "cpu",
    graph_def: Optional["CayleyGraphDef"] = None,
) -> tuple[nn.Module, ModelConfig]:
    """Loads model from a checkpoint written by :func:`save_checkpoint`.

    The returned model is in evaluation mode.

    :param path: Path to the checkpoint file.
    :param device: PyTorch device to load the model to.
    :param graph_def: Definition of the graph this model is going to be used with (optional). If given, and the
        checkpoint contains hash of the graph it was trained for, these graphs must be the same.
    :return: Pair (model, config describing this model).
    """
    # `weights_only=True` is passed explicitly: checkpoints contain only tensors and primitive values, so we never need
    # to unpickle arbitrary objects from them (and must not, because checkpoints can be downloaded from the internet).
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(
            f"File {path} is not a CayleyPy checkpoint (expected dict with keys 'config' and 'state_dict'). "
            "Bare state dicts must be loaded with ModelConfig.load."
        )
    format_version = int(checkpoint.get("format_version", 0))
    if format_version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has format version {format_version}, but this version of CayleyPy supports only "
            f"versions up to {CHECKPOINT_FORMAT_VERSION}. Please update CayleyPy."
        )
    config = ModelConfig.from_dict(checkpoint["config"])
    if graph_def is not None and config.graph_hash is not None:
        expected_hash = graph_hash(graph_def)
        if config.graph_hash != expected_hash:
            raise ValueError(
                f"Checkpoint {path} was trained for another graph (hash in checkpoint is {config.graph_hash}, "
                f"hash of the given graph is {expected_hash})."
            )
    model = config.build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.to(device), config
