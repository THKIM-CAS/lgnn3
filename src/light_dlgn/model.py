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


class InputWiseLogicLayer(nn.Module):
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.estimator = estimator
        self.residual_init = residual_init

        left = torch.randint(0, in_features, (out_features,), generator=generator)
        right = torch.randint(0, in_features, (out_features,), generator=generator)
        same = left == right
        if same.any():
            right[same] = (right[same] + 1) % in_features

        self.register_buffer("left_indices", left, persistent=True)
        self.register_buffer("right_indices", right, persistent=True)
        self.logits = nn.Parameter(torch.empty(out_features, 4))
        self.reset_parameters(generator=generator)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        with torch.no_grad():
            randn_kwargs = {"dtype": self.logits.dtype, "device": self.logits.device}
            if generator is not None:
                randn_kwargs["generator"] = generator

            if not self.residual_init:
                samples = torch.randn(self.logits.shape, **randn_kwargs)
                self.logits.copy_(samples)
                return

            mu, sigma = _heavy_tail_parameters(self.estimator)
            means = torch.tensor([-mu, -mu, mu, mu], dtype=self.logits.dtype, device=self.logits.device)
            samples = torch.randn(self.logits.shape, **randn_kwargs) * sigma + means
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

    def forward(self, x: torch.Tensor, *, discrete: bool = False) -> torch.Tensor:
        left = x.index_select(1, self.left_indices)
        right = x.index_select(1, self.right_indices)
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


class LightDLGN2(nn.Module):
    """Staged Light DLGN with per-class feedback reset."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        num_classes: int,
        steps_per_class: int,
        population: int,
        feedback_features: int,
        step_widths: tuple[int, ...],
        *,
        num_thresholds: int,
        tau: float,
        estimator: str = "sinusoidal",
        residual_init: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if steps_per_class <= 0:
            raise ValueError("steps_per_class must be > 0")
        if population <= 0:
            raise ValueError("population must be > 0")
        if feedback_features < 0:
            raise ValueError("feedback_features must be >= 0")
        if not step_widths:
            raise ValueError("step_widths must not be empty")

        encoded_dim = math.prod(image_shape) * num_thresholds
        if encoded_dim % steps_per_class != 0:
            raise ValueError("encoded input dimension must be divisible by steps_per_class")

        final_width = feedback_features + population
        if step_widths[-1] != final_width:
            raise ValueError("last step width must equal feedback_features + population")

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.steps_per_class = steps_per_class
        self.population = population
        self.feedback_features = feedback_features
        self.step_widths = tuple(step_widths)
        self.num_thresholds = num_thresholds
        self.tau = tau
        self.estimator = estimator
        self.residual_init = residual_init
        self.encoded_dim = encoded_dim
        self.partition_features = encoded_dim // steps_per_class

        self.register_buffer("thresholds", make_thresholds(num_thresholds), persistent=True)

        generator = torch.Generator()
        generator.manual_seed(seed)

        layers: list[nn.Module] = []
        in_features = self.partition_features + feedback_features
        for width in step_widths:
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

        self.step_layers = nn.ModuleList(layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return thermometer_encode(x, self.thresholds)

    def _step(self, x: torch.Tensor, *, discrete: bool) -> torch.Tensor:
        for layer in self.step_layers:
            x = layer(x, discrete=discrete)
        return x

    def forward(self, x: torch.Tensor, *, discrete: bool | None = None) -> torch.Tensor:
        if discrete is None:
            discrete = not self.training

        encoded = self.encode(x)
        partitions = encoded.split(self.partition_features, dim=1)
        logits: list[torch.Tensor] = []

        for _class_index in range(self.num_classes):
            feedback = encoded.new_zeros((encoded.size(0), self.feedback_features))
            population = encoded.new_empty((encoded.size(0), self.population))

            for partition in partitions:
                if self.feedback_features > 0:
                    step_input = torch.cat((partition, feedback), dim=1)
                else:
                    step_input = partition

                step_output = self._step(step_input, discrete=discrete)
                if self.feedback_features > 0:
                    feedback = step_output[:, : self.feedback_features]
                    population = step_output[:, self.feedback_features :]
                else:
                    population = step_output

            logits.append(population.sum(dim=1))

        return torch.stack(logits, dim=1) / self.tau
