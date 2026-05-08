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

Train the standard `LightDLGN` model.

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

## Train LightDLGN2

Train the staged `LightDLGN2` model with the corresponding training loop:

```bash
uv run train2.py --dataset mnist
uv run train2.py --dataset cifar10
```

`train2.py` uses the same dataset presets, optimizer defaults, validation metrics, and checkpoint cadence as `train.py`. The LightDLGN2-specific architecture options are:

- `--steps-per-class`: number of input partitions processed for each class, default `4`
- `--population`: output population per class, default `final profile width / num_classes`
- `--feedback-features`: feedback state carried between partitions, default equal to `population`
- `--step-widths`: comma-separated widths for the reused per-step logic stack

The final `--step-widths` value must equal `--feedback-features + --population`. If `--step-widths` is omitted, the script uses the dataset preset widths with the final width adjusted to satisfy that constraint.

Example smaller experimental run:

```bash
uv run train2.py \
  --dataset mnist \
  --epochs 5 \
  --population 256 \
  --feedback-features 64 \
  --step-widths 2048,320
```

LightDLGN2 artifacts are written separately to `artifacts/light_dlgn2/<dataset>/`:

- `best.pt`
- `last.pt`
- `history.json`

## Test

Evaluate a standard `LightDLGN` checkpoint on the official test split:

```bash
uv run test.py --checkpoint artifacts/mnist/best.pt
uv run test.py --checkpoint artifacts/cifar10/best.pt
```

Use `--mode continuous`, `--mode discrete`, or `--mode both`.

## Export Verilog

Export a trained discrete model as a clock-less Verilog-2005 combinational module:

```bash
uv run export_verilog.py \
  --checkpoint artifacts/paper/mnist/last.pt \
  --output artifacts/paper/mnist_logic.v \
  --module-name mnist_logic \
  --verify-samples 32
```

Generated module interface:

- `input [N-1:0] in_bits`
- `output [G-1:0] class0_bits`, ..., `class9_bits`

Usage notes:

- The Verilog module expects the input image to be already thermometer-encoded and flattened.
- Bit ordering matches `thermometer_encode()` in `src/light_dlgn/encoding.py`: the tensor is expanded along the threshold axis and then flattened.
- The exporter emits only `wire` declarations and `assign` statements.
- Each `class*_bits` bus is the raw final-layer bit group for that class. Popcount and argmax are intended to happen outside the generated module.
- `--verify-samples` runs a Python-side equivalence check between the exported discrete gate netlist and the PyTorch discrete model on random thermo-encoded inputs before writing the final file.

## Project Layout

```text
src/light_dlgn/model.py      # input-wise logic layers and classifier
src/light_dlgn/data.py       # MNIST/CIFAR-10 loaders
src/light_dlgn/config.py     # dataset presets
src/light_dlgn/export_verilog.py  # discrete netlist to Verilog exporter
train.py                     # training entry point
train2.py                    # LightDLGN2 training entry point
test.py                      # evaluation entry point
export_verilog.py            # Verilog export entry point
```

## Paper FCL Reference Models

The defaults in this repo are intentionally lighter for experimentation. The fully-connected reference models reported in the paper are:

| Dataset | Paper model details | Reported discretized test accuracy |
| --- | --- | --- |
| MNIST | 1-threshold thermometer encoding, followed by 4 `LogicLayerIWP` layers of width `32,000`, then `GroupSum(k=10, tau=100.0)` | `94.02 ± 0.08%` |
| CIFAR-10 | Baseline `CIFAR-10 M` FCL architecture reused from Petersen et al. (2022): 4 logic layers of width `128,000`, with `GroupSum(tau=100.0)` | `57.47 ± 0.20%` |

For context, the Light DLGN paper reports the corresponding original-parametrization baselines as:

- MNIST OP: `92.43 ± 0.17%`
- CIFAR-10 OP: `55.33 ± 0.23%`

Sources:

- Light DLGN Table 1 (reported discretized accuracies): https://openreview.net/pdf/206ddaa7c591e63de54dc2e750a9699f04ab5315.pdf
- Light DLGN Appendix D.4.2 and D.4.4 (MNIST/CIFAR-10 architecture notes): https://openreview.net/pdf/206ddaa7c591e63de54dc2e750a9699f04ab5315.pdf
- Original DLGN Appendix A.1, Table 6 (`CIFAR-10 medium` / `CIFAR-10 M`, `tau = 1/0.01 = 100`): https://proceedings.neurips.cc/paper_files/paper/2022/file/0d3496dd0cec77a999c98d35003203ca-Supplemental-Conference.pdf
- Official `difflogic` experiment commands (`mnist`, `cifar-10-3-thresholds`): https://github.com/Felix-Petersen/difflogic

## Reconstruct Reported Accuracy

The commands below match the paper settings as closely as this clean PyTorch reference supports:

- Adam with `lr=0.01`
- batch size `100`
- up to `200` epochs
- MNIST: 1-threshold encoding, width `32,000`, `tau=100`
- CIFAR-10: 3-threshold encoding, width `128,000`, 4 layers, `tau=100`
- no validation holdout: train on the full official training split, then evaluate on the official test split

MNIST:

```bash
uv run train.py \
  --dataset mnist \
  --epochs 200 \
  --batch-size 100 \
  --lr 0.01 \
  --widths 32000,32000,32000,32000 \
  --thresholds 1 \
  --tau 100 \
  --val-fraction 0 \
  --output-dir artifacts/paper

uv run test.py \
  --checkpoint artifacts/paper/mnist/last.pt \
  --mode discrete
```

CIFAR-10:

```bash
uv run train.py \
  --dataset cifar10 \
  --epochs 200 \
  --batch-size 100 \
  --lr 0.01 \
  --widths 128000,128000,128000,128000 \
  --thresholds 3 \
  --tau 100 \
  --val-fraction 0 \
  --output-dir artifacts/paper

uv run test.py \
  --checkpoint artifacts/paper/cifar10/last.pt \
  --mode discrete
```

Expected paper targets for the discretized model:

- MNIST: about `94.02 ± 0.08%`
- CIFAR-10: about `57.47 ± 0.20%`

This repo is a clean reimplementation rather than the original training code, so exact replication is not guaranteed. The commands above align the architecture, thresholds, temperature, optimizer, batch size, and training horizon with the paper-reported FCL settings.

## Notes

- The logic layer implements the paper's input-wise parametrization:
  - `g(p, q) = (1-p)(1-q)w00 + (1-p)q w01 + p(1-q)w10 + pq w11`
- The default estimator is the sinusoidal output estimator from the paper:
  - `w = 0.5 + 0.5 * sin(Omega)`
- Residual initialization biases each gate toward the pass-through `A` gate:
  - truth table `[0, 0, 1, 1]`
- This is a clean research reference, not a CUDA-optimized reproduction of the original `difflogic` codebase.
