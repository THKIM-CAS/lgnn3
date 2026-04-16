from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from light_dlgn.config import get_dataset_profile
from light_dlgn.data import build_test_loader
from light_dlgn.model import LightDLGN
from light_dlgn.train_utils import choose_device, evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Light DLGN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mode", choices=["continuous", "discrete", "both"], default="both")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    dataset_name = args.dataset or checkpoint["dataset"]
    profile = get_dataset_profile(dataset_name)
    model_cfg = checkpoint["model_config"]

    model = LightDLGN(
        image_shape=tuple(model_cfg["image_shape"]),
        num_classes=model_cfg["num_classes"],
        widths=tuple(model_cfg["widths"]),
        num_thresholds=model_cfg["num_thresholds"],
        tau=model_cfg["tau"],
        estimator=model_cfg["estimator"],
        residual_init=model_cfg["residual_init"],
        seed=model_cfg["seed"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    loader = build_test_loader(
        profile,
        root=str(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    criterion = nn.CrossEntropyLoss()

    if args.mode in {"continuous", "both"}:
        metrics = evaluate(model, loader, device=device, criterion=criterion, discrete=False)
        print(
            f"continuous test_loss={metrics['loss']:.4f} "
            f"test_acc={metrics['accuracy']:.4f}"
        )

    if args.mode in {"discrete", "both"}:
        metrics = evaluate(model, loader, device=device, criterion=criterion, discrete=True)
        print(
            f"discrete test_loss={metrics['loss']:.4f} "
            f"test_acc={metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
