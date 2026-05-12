from __future__ import annotations

import json
import random
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn


@lru_cache(maxsize=1)
def cuda_is_available() -> bool:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        available = torch.cuda.is_available()

    cuda_init_warnings = [
        warning for warning in caught_warnings if "CUDA initialization:" in str(warning.message)
    ]
    if not available and cuda_init_warnings:
        print(
            "CUDA is installed but unavailable to PyTorch. If this machine has an older NVIDIA driver, "
            "install a PyTorch build compiled for a compatible CUDA runtime, such as cu121."
        )
    return available


def reset_cuda_availability_cache() -> None:
    cuda_is_available.cache_clear()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if cuda_is_available() else "cpu")
    return torch.device(requested)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    discrete: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, discrete=discrete)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def init_wandb_run(config: dict[str, Any], *, project: str = "light-dlgn") -> Any | None:
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb is not installed; skipping wandb logging.")
        return None
    except Exception as exc:
        print(f"wandb could not be imported; skipping wandb logging: {exc}")
        return None

    try:
        return wandb.init(project=project, config=config)
    except Exception as exc:
        print(f"wandb initialization failed; skipping wandb logging: {exc}")
        return None


def log_wandb_metrics(run: Any | None, metrics: dict[str, float | int], *, step: int) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:
        print(f"wandb logging failed; continuing without blocking training: {exc}")


def finish_wandb_run(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"wandb finish failed: {exc}")
