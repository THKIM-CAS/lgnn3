from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    image_shape: tuple[int, int, int]
    num_classes: int
    widths: tuple[int, ...]
    thresholds: int
    tau: float
    batch_size: int
    epochs: int
    lr: float
    val_fraction: float = 0.1


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "mnist": DatasetProfile(
        name="mnist",
        image_shape=(1, 28, 28),
        num_classes=10,
        widths=(16_000, 16_000, 16_000, 16_000),
        thresholds=1,
        tau=100.0,
        batch_size=128,
        epochs=20,
        lr=1e-2,
    ),
    "cifar10": DatasetProfile(
        name="cifar10",
        image_shape=(3, 32, 32),
        num_classes=10,
        widths=(12_000, 12_000, 12_000, 12_000),
        thresholds=3,
        tau=100.0,
        batch_size=128,
        epochs=40,
        lr=1e-2,
    ),
}


def get_dataset_profile(name: str) -> DatasetProfile:
    try:
        return DATASET_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"unknown dataset '{name}', expected one of: {valid}") from exc
