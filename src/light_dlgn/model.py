from __future__ import annotations

import math

import torch
from torch import nn

from .encoding import make_thresholds, thermometer_encode


def _heavy_tail_parameters(estimator: str) -> tuple[float, float]:
    if estimator == "sinusoidal":
        return 1.2, 0.25
    if estimator == "sigmoid":
        return 3.0, 0.5
    raise ValueError(f"unsupported estimator '{estimator}'")


def _sample_connections(
    in_features: int, out_features: int, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced random wiring: every input is used floor/ceil(2*out/in) times, and no gate reads one input twice."""
    if in_features < 2:
        raise ValueError("in_features must be at least 2 to wire two distinct inputs per gate")

    n = 2 * out_features
    reps = -(-n // in_features)
    pool = torch.cat([torch.randperm(in_features, generator=generator) for _ in range(reps)])[:n]
    pool = pool[torch.randperm(n, generator=generator)]
    left, right = pool[:out_features].clone(), pool[out_features:].clone()

    # Repair the few self-connections by swapping right inputs with other gates; swaps keep usage counts balanced.
    for i in (left == right).nonzero().flatten().tolist():
        while left[i] == right[i]:
            j = int(torch.randint(0, out_features, (1,), generator=generator))
            if left[j] != right[i] and left[i] != right[j]:
                right[i], right[j] = right[j].clone(), right[i].clone()
    return left, right


class _TruthTableGates(nn.Module):
    """Trainable two-input truth tables shared by logic-layer implementations."""

    def __init__(
        self,
        out_features: int,
        *,
        estimator: str = "sinusoidal",
        residual_init: bool = True,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.estimator = estimator
        self.residual_init = residual_init
        self.logits = nn.Parameter(torch.empty(out_features, 4))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if not self.residual_init:
                nn.init.normal_(self.logits, mean=0.0, std=1.0)
                return

            mu, sigma = _heavy_tail_parameters(self.estimator)
            means = torch.tensor([-mu, -mu, mu, mu], dtype=self.logits.dtype, device=self.logits.device)
            samples = torch.randn_like(self.logits) * sigma + means
            self.logits.copy_(samples)

    def _coefficients(self, discrete: bool) -> torch.Tensor:
        if self.estimator == "sinusoidal":
            omega = 0.5 + 0.5 * torch.sin(self.logits)
        elif self.estimator == "sigmoid":
            omega = torch.sigmoid(self.logits)
        else:
            raise ValueError(f"unsupported estimator '{self.estimator}'")

        if discrete:
            return (omega > 0.5).to(dtype=omega.dtype)
        return omega

    def forward(self, left: torch.Tensor, right: torch.Tensor, *, discrete: bool = False) -> torch.Tensor:
        omega = self._coefficients(discrete)
        w00 = omega[:, 0].unsqueeze(0)
        w01 = omega[:, 1].unsqueeze(0)
        w10 = omega[:, 2].unsqueeze(0)
        w11 = omega[:, 3].unsqueeze(0)
        return (
            (1.0 - left) * (1.0 - right) * w00
            + (1.0 - left) * right * w01
            + left * (1.0 - right) * w10
            + left * right * w11
        )


class InputWiseLogicLayer(_TruthTableGates):
    """Binary-input DLGN layer using the paper's input-wise parametrization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        estimator: str = "sinusoidal",
        residual_init: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(
            out_features,
            estimator=estimator,
            residual_init=residual_init,
        )
        self.in_features = in_features

        left, right = _sample_connections(in_features, out_features, generator)
        self.register_buffer("left_indices", left, persistent=True)
        self.register_buffer("right_indices", right, persistent=True)

    def forward(self, x: torch.Tensor, *, discrete: bool = False) -> torch.Tensor:
        left = x.index_select(1, self.left_indices)
        right = x.index_select(1, self.right_indices)
        return super().forward(left, right, discrete=discrete)


class LogicTree(nn.Module):
    """Tournament-style logic tree with learned gates and sports-style byes.

    ``out_features`` must be reachable by repeatedly pairing adjacent values and
    advancing an unpaired final value to the next round.  For example, a tree
    with five inputs can expose widths 5, 3, 2, or 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        estimator: str = "sinusoidal",
        residual_init: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must both be at least 1")

        reachable_widths = [in_features]
        width = in_features
        while width > 1:
            width = (width + 1) // 2
            reachable_widths.append(width)
        if out_features not in reachable_widths:
            choices = ", ".join(str(value) for value in reachable_widths)
            raise ValueError(
                f"out_features={out_features} is not reachable from in_features={in_features}; "
                f"choose one of: {choices}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.estimator = estimator
        self.residual_init = residual_init

        levels: list[nn.Module] = []
        width = in_features
        while width != out_features:
            levels.append(
                _TruthTableGates(
                    width // 2,
                    estimator=estimator,
                    residual_init=residual_init,
                )
            )
            width = (width + 1) // 2
        self.logic_levels = nn.ModuleList(levels)

    def forward(self, x: torch.Tensor, *, discrete: bool = False) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"expected x with shape (batch_size, {self.in_features}), got {tuple(x.shape)}"
            )

        values = x
        for level in self.logic_levels:
            paired_width = 2 * level.out_features
            gates = level(
                values[:, :paired_width:2],
                values[:, 1:paired_width:2],
                discrete=discrete,
            )
            if values.size(1) % 2:
                values = torch.cat((gates, values[:, -1:]), dim=1)
            else:
                values = gates
        return values


class GroupSum(nn.Module):
    def __init__(self, num_classes: int, tau: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) % self.num_classes != 0:
            raise ValueError(
                f"final width {x.size(1)} must be divisible by num_classes={self.num_classes}"
            )
        grouped = x.view(x.size(0), self.num_classes, -1)
        return grouped.sum(dim=-1) / self.tau


class LightDLGN(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        num_classes: int,
        widths: tuple[int, ...],
        *,
        num_thresholds: int,
        tau: float,
        estimator: str = "sinusoidal",
        residual_init: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths must not be empty")
        if widths[-1] % num_classes != 0:
            raise ValueError("last width must be divisible by num_classes")

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.num_thresholds = num_thresholds
        self.tau = tau
        self.estimator = estimator
        self.residual_init = residual_init

        encoded_dim = math.prod(image_shape) * num_thresholds
        self.register_buffer("thresholds", make_thresholds(num_thresholds), persistent=True)

        generator = torch.Generator()
        generator.manual_seed(seed)

        layers: list[nn.Module] = []
        in_features = encoded_dim
        for width in widths:
            layers.append(
                InputWiseLogicLayer(
                    in_features,
                    width,
                    estimator=estimator,
                    residual_init=residual_init,
                    generator=generator,
                )
            )
            in_features = width

        self.logic_layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes=num_classes, tau=tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return thermometer_encode(x, self.thresholds)

    def forward(self, x: torch.Tensor, *, discrete: bool | None = None) -> torch.Tensor:
        if discrete is None:
            discrete = not self.training
        x = self.encode(x)
        for layer in self.logic_layers:
            x = layer(x, discrete=discrete)
        return self.group_sum(x)
