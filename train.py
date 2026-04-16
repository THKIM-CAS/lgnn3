from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from light_dlgn.config import get_dataset_profile
from light_dlgn.data import build_dataloaders
from light_dlgn.model import build_model
from light_dlgn.train_utils import choose_device, evaluate, save_checkpoint, save_history, seed_everything


def parse_widths(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fully-connected Light DLGN.")
    parser.add_argument("--model", choices=["lightdlgn", "multiplexed"], default="lightdlgn")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--widths", type=str, default=None, help="comma-separated layer widths")
    parser.add_argument("--thresholds", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--estimator", choices=["sinusoidal", "sigmoid"], default="sinusoidal")
    parser.add_argument("--disable-residual-init", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = get_dataset_profile(args.dataset)
    seed_everything(args.seed)
    device = choose_device(args.device)

    widths = parse_widths(args.widths, profile.widths)
    thresholds = args.thresholds if args.thresholds is not None else profile.thresholds
    tau = args.tau if args.tau is not None else profile.tau
    epochs = args.epochs if args.epochs is not None else profile.epochs
    batch_size = args.batch_size if args.batch_size is not None else profile.batch_size
    lr = args.lr if args.lr is not None else profile.lr
    val_fraction = args.val_fraction if args.val_fraction is not None else profile.val_fraction

    loaders = build_dataloaders(
        profile,
        root=str(args.data_dir),
        batch_size=batch_size,
        val_fraction=val_fraction,
        num_workers=args.workers,
        seed=args.seed,
    )

    model = build_model(
        args.model,
        image_shape=profile.image_shape,
        num_classes=profile.num_classes,
        widths=widths,
        num_thresholds=thresholds,
        tau=tau,
        estimator=args.estimator,
        residual_init=not args.disable_residual_init,
        seed=args.seed,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    run_dir = args.output_dir / args.dataset
    if args.model != "lightdlgn":
        run_dir = args.output_dir / args.model / args.dataset
    history: list[dict] = []
    best_discrete_val = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        progress = tqdm(loaders.train, desc=f"epoch {epoch}/{epochs}", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, discrete=False)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size_now = targets.size(0)
            running_loss += loss.item() * batch_size_now
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_examples += batch_size_now
            progress.set_postfix(
                loss=f"{running_loss / running_examples:.4f}",
                acc=f"{running_correct / running_examples:.4f}",
            )

        train_metrics = {
            "loss": running_loss / running_examples,
            "accuracy": running_correct / running_examples,
        }
        if loaders.val is None:
            val_continuous = None
            val_discrete = None
        else:
            val_continuous = evaluate(
                model,
                loaders.val,
                device=device,
                criterion=criterion,
                discrete=False,
            )
            val_discrete = evaluate(
                model,
                loaders.val,
                device=device,
                criterion=criterion,
                discrete=True,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val_continuous": val_continuous,
            "val_discrete": val_discrete,
        }
        history.append(epoch_metrics)

        if val_continuous is None or val_discrete is None:
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )
        else:
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"val_cont_loss={val_continuous['loss']:.4f} val_cont_acc={val_continuous['accuracy']:.4f} "
                f"val_disc_loss={val_discrete['loss']:.4f} val_disc_acc={val_discrete['accuracy']:.4f}"
            )

        checkpoint = {
            "dataset": args.dataset,
            "model_config": {
                "model_type": args.model,
                "image_shape": profile.image_shape,
                "num_classes": profile.num_classes,
                "widths": widths,
                "num_thresholds": thresholds,
                "tau": tau,
                "estimator": args.estimator,
                "residual_init": not args.disable_residual_init,
                "seed": args.seed,
            },
            "train_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "val_fraction": val_fraction,
            },
            "epoch": epoch,
            "history": history,
            "model_state": model.state_dict(),
        }
        save_checkpoint(run_dir / "last.pt", checkpoint)

        metric_for_best = train_metrics["accuracy"] if val_discrete is None else val_discrete["accuracy"]
        if metric_for_best > best_discrete_val:
            best_discrete_val = metric_for_best
            save_checkpoint(run_dir / "best.pt", checkpoint)

    save_history(run_dir / "history.json", history)


if __name__ == "__main__":
    main()
