# Light DLGN

Clean PyTorch reference implementation of a fully-connected Light DLGN based on the input-wise parametrization from "Light Differentiable Logic Gate Networks".

This project focuses on:

- `uv`-managed Python setup
- PyTorch implementation only
- fully-connected models only
- MNIST and CIFAR-10 presets
- separate model, train, and test code

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Train

MNIST:

```bash
uv run train.py --dataset mnist
```

CIFAR-10:

```bash
uv run train.py --dataset cifar10
```

Useful overrides:

```bash
uv run train.py --dataset cifar10 --widths 12000,12000,12000,12000 --epochs 50
uv run train.py --dataset mnist --estimator sigmoid
```

Artifacts are written to `artifacts/<dataset>/`:

- `best.pt`
- `last.pt`
- `history.json`

## Test

Evaluate the best checkpoint on the official test split:

```bash
uv run test.py --checkpoint artifacts/mnist/best.pt
uv run test.py --checkpoint artifacts/cifar10/best.pt
```

Use `--mode continuous`, `--mode discrete`, or `--mode both`.

## Project Layout

```text
src/light_dlgn/model.py      # input-wise logic layers and classifier
src/light_dlgn/data.py       # MNIST/CIFAR-10 loaders
src/light_dlgn/config.py     # dataset presets
train.py                     # training entry point
test.py                      # evaluation entry point
```

## Notes

- The logic layer implements the paper's input-wise parametrization:
  - `g(p, q) = (1-p)(1-q)w00 + (1-p)q w01 + p(1-q)w10 + pq w11`
- The default estimator is the sinusoidal output estimator from the paper:
  - `w = 0.5 + 0.5 * sin(Omega)`
- Residual initialization biases each gate toward the pass-through `A` gate:
  - truth table `[0, 0, 1, 1]`
- This is a clean research reference, not a CUDA-optimized reproduction of the original `difflogic` codebase.
