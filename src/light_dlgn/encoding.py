from __future__ import annotations

import torch


def make_thresholds(num_thresholds: int) -> torch.Tensor:
    if num_thresholds < 1:
        raise ValueError("num_thresholds must be >= 1")
    return torch.linspace(0.0, 1.0, steps=num_thresholds + 2, dtype=torch.float32)[1:-1]


def thermometer_encode(x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"expected BCHW input tensor, got shape {tuple(x.shape)}")
    thresholds = thresholds.to(device=x.device, dtype=x.dtype)
    bits = (x.unsqueeze(-1) >= thresholds.view(1, 1, 1, 1, -1)).to(x.dtype)
    return bits.flatten(1)
