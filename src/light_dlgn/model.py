from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .encoding import make_thresholds, thermometer_encode


def _heavy_tail_parameters(estimator: str) -> tuple[float, float]:
    if estimator == "sinusoidal":
        return 1.2, 0.25
    if estimator == "sigmoid":
        return 3.0, 0.5
    raise ValueError(f"unsupported estimator '{estimator}'")


def _apply_estimator(logits: torch.Tensor, *, estimator: str, discrete: bool) -> torch.Tensor:
    if estimator == "sinusoidal":
        omega = 0.5 + 0.5 * torch.sin(logits)
    elif estimator == "sigmoid":
        omega = torch.sigmoid(logits)
    else:
        raise ValueError(f"unsupported estimator '{estimator}'")

    if discrete:
        return (omega > 0.5).to(dtype=omega.dtype)
    return omega


def _make_fixed_binary_class_codes(
    *,
    num_layers: int,
    num_classes: int,
    code_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    bits_needed = max(1, (num_classes - 1).bit_length())
    if code_dim < bits_needed:
        raise ValueError(
            f"class-code width {code_dim} is too small for {num_classes} classes; "
            f"need at least {bits_needed} bits for distinct fixed codes"
        )

    codes = torch.randint(
        0,
        2,
        (num_layers, num_classes, code_dim),
        generator=generator,
        dtype=torch.float32,
    )

    class_ids = torch.arange(num_classes, dtype=torch.long)
    bit_positions = torch.arange(bits_needed, dtype=torch.long)
    id_bits = ((class_ids.unsqueeze(1) >> bit_positions) & 1).to(torch.float32)
    codes[:, :, :bits_needed] = id_bits.unsqueeze(0)
    return codes


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
        return _apply_estimator(self.logits, estimator=self.estimator, discrete=discrete)

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


class ClassConditionedInputWiseLogicLayer(nn.Module):
    """Input-wise logic layer with some gates explicitly conditioned on class code."""

    def __init__(
        self,
        data_features: int,
        code_features: int,
        out_features: int,
        *,
        code_gate_fraction: float = 0.5,
        estimator: str = "sinusoidal",
        residual_init: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if data_features <= 0:
            raise ValueError("data_features must be positive")
        if code_features <= 0:
            raise ValueError("code_features must be positive")
        if not 0.0 <= code_gate_fraction <= 1.0:
            raise ValueError("code_gate_fraction must be in [0, 1]")

        self.data_features = data_features
        self.code_features = code_features
        self.in_features = data_features + code_features
        self.out_features = out_features
        self.code_gate_fraction = code_gate_fraction
        self.estimator = estimator
        self.residual_init = residual_init

        if code_gate_fraction == 0.0:
            forced_count = 0
        else:
            forced_count = max(1, math.ceil(out_features * code_gate_fraction))
        self.forced_code_gate_count = forced_count

        left = torch.randint(0, data_features, (out_features,), generator=generator)
        right = torch.randint(0, data_features, (out_features,), generator=generator)
        same = left == right
        if same.any() and data_features > 1:
            right[same] = (right[same] + 1) % data_features

        self.register_buffer("left_indices", left, persistent=True)
        self.register_buffer("right_indices", right, persistent=True)

        forced_mask = torch.zeros(out_features, dtype=torch.bool)
        if forced_count:
            forced_mask[:forced_count] = True
        code_indices = torch.randint(0, code_features, (forced_count,), generator=generator)
        self.register_buffer("forced_code_gate_mask", forced_mask, persistent=True)
        self.register_buffer("code_indices", code_indices, persistent=True)

        self.logits = nn.Parameter(torch.empty(out_features, 4))
        self.code_logits = nn.Parameter(torch.empty(forced_count, 4))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if not self.residual_init:
                nn.init.normal_(self.logits, mean=0.0, std=1.0)
                nn.init.normal_(self.code_logits, mean=0.0, std=1.0)
                return

            mu, sigma = _heavy_tail_parameters(self.estimator)
            means = torch.tensor([-mu, -mu, mu, mu], dtype=self.logits.dtype, device=self.logits.device)
            self.logits.copy_(torch.randn_like(self.logits) * sigma + means)
            self.code_logits.copy_(torch.randn_like(self.code_logits) * sigma + means)

    def _coefficients(self, discrete: bool) -> torch.Tensor:
        return _apply_estimator(self.logits, estimator=self.estimator, discrete=discrete)

    def _code_coefficients(self, discrete: bool) -> torch.Tensor:
        return _apply_estimator(self.code_logits, estimator=self.estimator, discrete=discrete)

    @staticmethod
    def _apply_binary_truth_table(
        left: torch.Tensor,
        right: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, *, discrete: bool = False) -> torch.Tensor:
        data = x[:, : self.data_features]
        code = x[:, self.data_features :]

        left = data.index_select(1, self.left_indices)
        right = data.index_select(1, self.right_indices)
        values = self._apply_binary_truth_table(left, right, self._coefficients(discrete))

        if self.forced_code_gate_count:
            count = self.forced_code_gate_count
            conditioned = self._apply_binary_truth_table(
                left[:, :count],
                right[:, :count],
                self._code_coefficients(discrete),
            )
            selected_code = code.index_select(1, self.code_indices)
            blended = (1.0 - selected_code) * values[:, :count] + selected_code * conditioned
            if count == self.out_features:
                values = blended
            else:
                values = torch.cat((blended, values[:, count:]), dim=1)

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


class MultiplexedLightDLGN(nn.Module):
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
        if len(widths) < 2:
            raise ValueError(
                "multiplexed widths must include the class-code width followed by at least one logic-layer width"
            )
        if widths[0] <= 0:
            raise ValueError("class-code width must be positive")

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.class_code_dim = widths[0]
        self.shared_widths = tuple(widths[1:])
        self.num_logic_layers = len(self.shared_widths)
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
        for width in self.shared_widths:
            layers.append(
                InputWiseLogicLayer(
                    in_features + self.class_code_dim,
                    width,
                    estimator=estimator,
                    residual_init=residual_init,
                    generator=generator,
                )
            )
            in_features = width

        self.class_code_logits = nn.Parameter(
            torch.empty(self.num_logic_layers, num_classes, self.class_code_dim)
        )
        self.logic_layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes=1, tau=tau)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.class_code_logits, mean=0.0, std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return thermometer_encode(x, self.thresholds)

    def class_codes(self, *, discrete: bool) -> torch.Tensor:
        return _apply_estimator(self.class_code_logits, estimator=self.estimator, discrete=discrete)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        class_code_logits = state_dict.get("class_code_logits")
        if isinstance(class_code_logits, torch.Tensor) and class_code_logits.dim() == 2:
            raise RuntimeError(
                "legacy multiplexed checkpoints are incompatible with the current architecture: "
                "expected per-layer class_code_logits with shape "
                "[num_logic_layers, num_classes, class_code_dim], but found the previous "
                "single-matrix format [num_classes, class_code_dim]"
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor, *, discrete: bool | None = None) -> torch.Tensor:
        if discrete is None:
            discrete = not self.training

        encoded = self.encode(x)
        codes = self.class_codes(discrete=discrete)

        batch_size = encoded.size(0)
        values = encoded
        for layer_index, layer in enumerate(self.logic_layers):
            if values.dim() == 2:
                current_values = values.unsqueeze(1).expand(-1, self.num_classes, -1)
            else:
                current_values = values
            current_codes = codes[layer_index].unsqueeze(0).expand(batch_size, -1, -1)
            layer_inputs = torch.cat((current_values, current_codes), dim=-1)
            layer_inputs = layer_inputs.reshape(batch_size * self.num_classes, -1)
            values = layer(layer_inputs, discrete=discrete)
            if layer_index + 1 < self.num_logic_layers:
                values = values.view(batch_size, self.num_classes, -1)

        logits = self.group_sum(values)
        return logits.view(batch_size, self.num_classes)


class MultiplexedLightDLGN2(nn.Module):
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
        code_gate_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError(
                "multiplexed2 widths must include the class-code width followed by at least one logic-layer width"
            )
        if widths[0] <= 0:
            raise ValueError("class-code width must be positive")

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.widths = tuple(widths)
        self.class_code_dim = widths[0]
        self.shared_widths = tuple(widths[1:])
        self.num_logic_layers = len(self.shared_widths)
        self.num_thresholds = num_thresholds
        self.tau = tau
        self.estimator = estimator
        self.residual_init = residual_init
        self.code_gate_fraction = code_gate_fraction

        encoded_dim = math.prod(image_shape) * num_thresholds
        self.register_buffer("thresholds", make_thresholds(num_thresholds), persistent=True)

        generator = torch.Generator()
        generator.manual_seed(seed)

        code_generator = torch.Generator()
        code_generator.manual_seed(seed + 1)
        self.register_buffer(
            "fixed_class_codes",
            _make_fixed_binary_class_codes(
                num_layers=self.num_logic_layers,
                num_classes=num_classes,
                code_dim=self.class_code_dim,
                generator=code_generator,
            ),
            persistent=True,
        )

        layers: list[nn.Module] = []
        in_features = encoded_dim
        for width in self.shared_widths:
            layers.append(
                ClassConditionedInputWiseLogicLayer(
                    in_features,
                    self.class_code_dim,
                    width,
                    code_gate_fraction=code_gate_fraction,
                    estimator=estimator,
                    residual_init=residual_init,
                    generator=generator,
                )
            )
            in_features = width

        self.logic_layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes=1, tau=tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return thermometer_encode(x, self.thresholds)

    def class_codes(self, *, discrete: bool) -> torch.Tensor:
        return self.fixed_class_codes

    def forward(self, x: torch.Tensor, *, discrete: bool | None = None) -> torch.Tensor:
        if discrete is None:
            discrete = not self.training

        encoded = self.encode(x)
        codes = self.class_codes(discrete=discrete)

        batch_size = encoded.size(0)
        values = encoded
        for layer_index, layer in enumerate(self.logic_layers):
            if values.dim() == 2:
                current_values = values.unsqueeze(1).expand(-1, self.num_classes, -1)
            else:
                current_values = values
            current_codes = codes[layer_index].unsqueeze(0).expand(batch_size, -1, -1)
            layer_inputs = torch.cat((current_values, current_codes), dim=-1)
            layer_inputs = layer_inputs.reshape(batch_size * self.num_classes, -1)
            values = layer(layer_inputs, discrete=discrete)
            if layer_index + 1 < self.num_logic_layers:
                values = values.view(batch_size, self.num_classes, -1)

        logits = self.group_sum(values)
        return logits.view(batch_size, self.num_classes)


def build_model(
    model_type: str,
    *,
    image_shape: tuple[int, int, int],
    num_classes: int,
    widths: tuple[int, ...],
    num_thresholds: int,
    tau: float,
    estimator: str = "sinusoidal",
    residual_init: bool = True,
    seed: int = 0,
    code_gate_fraction: float = 0.5,
) -> nn.Module:
    kwargs = {
        "image_shape": image_shape,
        "num_classes": num_classes,
        "widths": widths,
        "num_thresholds": num_thresholds,
        "tau": tau,
        "estimator": estimator,
        "residual_init": residual_init,
        "seed": seed,
    }
    if model_type == "lightdlgn":
        return LightDLGN(**kwargs)
    if model_type == "multiplexed":
        return MultiplexedLightDLGN(**kwargs)
    if model_type == "multiplexed2":
        return MultiplexedLightDLGN2(**kwargs, code_gate_fraction=code_gate_fraction)
    raise ValueError(f"unsupported model_type '{model_type}'")
