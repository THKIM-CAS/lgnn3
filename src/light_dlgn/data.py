from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DatasetProfile


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _dataset_factory(name: str):
    if name == "mnist":
        return datasets.MNIST
    if name == "cifar10":
        return datasets.CIFAR10
    raise ValueError(f"unsupported dataset '{name}'")


def _transforms(name: str) -> tuple[transforms.Compose, transforms.Compose]:
    if name == "mnist":
        tensor_only = transforms.Compose([transforms.ToTensor()])
        return tensor_only, tensor_only

    if name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor()])
        return train_transform, eval_transform

    raise ValueError(f"unsupported dataset '{name}'")


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_dataloaders(
    profile: DatasetProfile,
    *,
    root: str,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    seed: int,
) -> DataLoaders:
    dataset_cls = _dataset_factory(profile.name)
    train_transform, eval_transform = _transforms(profile.name)

    full_train_aug = dataset_cls(root=root, train=True, download=True, transform=train_transform)
    full_train_eval = dataset_cls(root=root, train=True, download=True, transform=eval_transform)
    test_set = dataset_cls(root=root, train=False, download=True, transform=eval_transform)

    total = len(full_train_aug)
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("validation fraction leaves no training samples")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    return DataLoaders(
        train=_make_loader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        val=_make_loader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=_make_loader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def build_test_loader(
    profile: DatasetProfile,
    *,
    root: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset_cls = _dataset_factory(profile.name)
    _, eval_transform = _transforms(profile.name)
    test_set = dataset_cls(root=root, train=False, download=True, transform=eval_transform)
    return _make_loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
