import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Union

import torch


@dataclass(frozen=True)
class DeviceConfig:
    """Normalized device configuration for CayleyGraph."""

    devices: tuple[torch.device, ...]

    @staticmethod
    def create(
        device: str = "auto",
        num_gpus: Optional[int] = None,
        specific_devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
    ) -> "DeviceConfig":
        return DeviceConfig(DeviceConfig._resolve_devices(device, num_gpus, specific_devices))

    @cached_property
    def device(self) -> torch.device:
        return self.devices[0]

    @cached_property
    def gpu_devices(self) -> list[torch.device]:
        return [device for device in self.devices if device.type == "cuda"]

    @cached_property
    def num_gpus(self) -> int:
        return len(self.gpu_devices)

    @staticmethod
    def _normalize_cuda_device(device: Union[int, str, torch.device]) -> torch.device:
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        normalized = torch.device(device)
        if normalized.type == "cuda" and normalized.index is None:
            return torch.device("cuda:0")
        return normalized

    @staticmethod
    def _resolve_devices(
        device: str,
        num_gpus: Optional[int],
        specific_devices: Optional[Sequence[Union[int, str, torch.device]]],
    ) -> tuple[torch.device, ...]:
        if specific_devices is not None:
            resolved = tuple(DeviceConfig._normalize_cuda_device(dev) for dev in specific_devices)
            if not resolved:
                raise ValueError("specific_devices must not be empty.")
            if not torch.cuda.is_available():
                raise ValueError("specific_devices requires CUDA, but CUDA is not available.")
            available = torch.cuda.device_count()
            for cuda_device in resolved:
                if cuda_device.type != "cuda":
                    raise ValueError("specific_devices must contain only CUDA devices.")
                if cuda_device.index is None or cuda_device.index >= available:
                    raise ValueError(f"CUDA device {cuda_device} is not available.")
            return resolved

        if device == "gpu":
            device = "cuda"
        if device not in ["auto", "cpu", "cuda"]:
            raise ValueError("device must be one of 'auto', 'cpu', 'cuda', or 'gpu'.")

        if device == "cpu":
            if num_gpus not in [None, 0, 1]:
                raise ValueError("device='cpu' only supports num_gpus=None, 0, or 1.")
            if num_gpus == 1:
                warnings.warn("num_gpus=1 was provided with device='cpu'; using the CPU single-device path.")
            return (torch.device("cpu"),)

        if device == "auto":
            if num_gpus == 0:
                return (torch.device("cpu"),)
            if not torch.cuda.is_available():
                if num_gpus not in [None, 1]:
                    raise ValueError("num_gpus was requested, but CUDA is not available.")
                return (torch.device("cpu"),)
            device = "cuda"

        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but CUDA is not available.")

        available = torch.cuda.device_count()
        if num_gpus is None:
            resolved_num_gpus = available
        else:
            if num_gpus <= 0:
                raise ValueError("num_gpus must be positive when CUDA is selected.")
            if num_gpus > available:
                raise ValueError(f"Requested {num_gpus} GPUs, but only {available} are available.")
            resolved_num_gpus = num_gpus

        return tuple(torch.device(f"cuda:{i}") for i in range(resolved_num_gpus))
