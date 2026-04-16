from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import InputWiseLogicLayer, LightDLGN


@dataclass(frozen=True)
class DiscreteGate:
    left_index: int
    right_index: int
    truth_table: tuple[int, int, int, int]


@dataclass(frozen=True)
class LogicNetlist:
    encoded_dim: int
    layer_widths: tuple[int, ...]
    num_classes: int
    gates_by_layer: tuple[tuple[DiscreteGate, ...], ...]

    @property
    def group_size(self) -> int:
        return self.layer_widths[-1] // self.num_classes


_TRUTH_TABLE_TO_EXPR: dict[tuple[int, int, int, int], str] = {
    (0, 0, 0, 0): "1'b0",
    (0, 0, 0, 1): "({a} & {b})",
    (0, 0, 1, 0): "({a} & ~{b})",
    (0, 0, 1, 1): "{a}",
    (0, 1, 0, 0): "(~{a} & {b})",
    (0, 1, 0, 1): "{b}",
    (0, 1, 1, 0): "({a} ^ {b})",
    (0, 1, 1, 1): "({a} | {b})",
    (1, 0, 0, 0): "~({a} | {b})",
    (1, 0, 0, 1): "~({a} ^ {b})",
    (1, 0, 1, 0): "~{b}",
    (1, 0, 1, 1): "({a} | ~{b})",
    (1, 1, 0, 0): "~{a}",
    (1, 1, 0, 1): "(~{a} | {b})",
    (1, 1, 1, 0): "~({a} & {b})",
    (1, 1, 1, 1): "1'b1",
}


def load_model_from_checkpoint(checkpoint_path: Path, *, device: str = "cpu") -> LightDLGN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _layer_to_discrete_gates(layer: InputWiseLogicLayer) -> tuple[DiscreteGate, ...]:
    omega = layer._coefficients(discrete=True).to(dtype=torch.int64).cpu()
    left_indices = layer.left_indices.cpu().tolist()
    right_indices = layer.right_indices.cpu().tolist()
    gates: list[DiscreteGate] = []
    for gate_index in range(layer.out_features):
        truth_table = tuple(int(bit) for bit in omega[gate_index].tolist())
        gates.append(
            DiscreteGate(
                left_index=int(left_indices[gate_index]),
                right_index=int(right_indices[gate_index]),
                truth_table=truth_table,
            )
        )
    return tuple(gates)


def extract_logic_netlist(model: LightDLGN) -> LogicNetlist:
    gates_by_layer = tuple(_layer_to_discrete_gates(layer) for layer in model.logic_layers)
    return LogicNetlist(
        encoded_dim=math.prod(model.image_shape) * model.num_thresholds,
        layer_widths=tuple(model.widths),
        num_classes=model.num_classes,
        gates_by_layer=gates_by_layer,
    )


def _format_gate_expr(truth_table: tuple[int, int, int, int], *, left_expr: str, right_expr: str) -> str:
    try:
        template = _TRUTH_TABLE_TO_EXPR[truth_table]
    except KeyError as exc:
        raise ValueError(f"unsupported truth table {truth_table}") from exc
    return template.format(a=left_expr, b=right_expr)


def netlist_to_verilog(netlist: LogicNetlist, *, module_name: str) -> str:
    lines: list[str] = []
    lines.append(f"module {module_name}(")
    lines.append("    in_bits,")
    for class_index in range(netlist.num_classes):
        suffix = "," if class_index + 1 < netlist.num_classes else ""
        lines.append(f"    class{class_index}_bits{suffix}")
    lines.append(");")
    lines.append("")
    lines.append(f"input [{netlist.encoded_dim - 1}:0] in_bits;")
    for class_index in range(netlist.num_classes):
        lines.append(f"output [{netlist.group_size - 1}:0] class{class_index}_bits;")
    lines.append("")

    previous_refs = [f"in_bits[{index}]" for index in range(netlist.encoded_dim)]
    for layer_index, layer_gates in enumerate(netlist.gates_by_layer):
        for neuron_index, gate in enumerate(layer_gates):
            lines.append(f"wire l{layer_index}_{neuron_index};")
        lines.append("")
        current_refs: list[str] = []
        for neuron_index, gate in enumerate(layer_gates):
            wire_name = f"l{layer_index}_{neuron_index}"
            left_expr = previous_refs[gate.left_index]
            right_expr = previous_refs[gate.right_index]
            gate_expr = _format_gate_expr(gate.truth_table, left_expr=left_expr, right_expr=right_expr)
            lines.append(f"assign {wire_name} = {gate_expr};")
            current_refs.append(wire_name)
        lines.append("")
        previous_refs = current_refs

    for class_index in range(netlist.num_classes):
        start = class_index * netlist.group_size
        for offset in range(netlist.group_size):
            lines.append(f"assign class{class_index}_bits[{offset}] = {previous_refs[start + offset]};")
        lines.append("")

    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def _apply_truth_table(
    truth_table: tuple[int, int, int, int],
    left_values: torch.Tensor,
    right_values: torch.Tensor,
) -> torch.Tensor:
    tt = truth_table
    return (
        ((1 - left_values) & (1 - right_values) & tt[0])
        | ((1 - left_values) & right_values & tt[1])
        | (left_values & (1 - right_values) & tt[2])
        | (left_values & right_values & tt[3])
    )


def evaluate_netlist_bits(netlist: LogicNetlist, encoded_inputs: torch.Tensor) -> torch.Tensor:
    values = encoded_inputs.to(dtype=torch.int64)
    for layer_gates in netlist.gates_by_layer:
        next_values = torch.empty((values.size(0), len(layer_gates)), dtype=torch.int64)
        for neuron_index, gate in enumerate(layer_gates):
            left_values = values[:, gate.left_index]
            right_values = values[:, gate.right_index]
            next_values[:, neuron_index] = _apply_truth_table(gate.truth_table, left_values, right_values)
        values = next_values
    return values


def verify_netlist(
    model: LightDLGN,
    netlist: LogicNetlist,
    *,
    samples: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    encoded_inputs = torch.randint(0, 2, (samples, netlist.encoded_dim), generator=generator, dtype=torch.int64)
    netlist_outputs = evaluate_netlist_bits(netlist, encoded_inputs)

    with torch.no_grad():
        values = encoded_inputs.to(dtype=torch.float32)
        for layer in model.logic_layers:
            values = layer(values, discrete=True)
        model_outputs = values.to(dtype=torch.int64)

    if not torch.equal(netlist_outputs, model_outputs):
        mismatch = (netlist_outputs != model_outputs).nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            "exported netlist does not match discrete model output "
            f"at sample={mismatch[0]} bit={mismatch[1]}"
        )


def write_verilog_module(
    checkpoint_path: Path,
    output_path: Path,
    *,
    module_name: str,
    verify_samples: int = 0,
    verify_seed: int = 0,
) -> LogicNetlist:
    model = load_model_from_checkpoint(checkpoint_path)
    netlist = extract_logic_netlist(model)
    if verify_samples > 0:
        verify_netlist(model, netlist, samples=verify_samples, seed=verify_seed)
    verilog = netlist_to_verilog(netlist, module_name=module_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(verilog, encoding="utf-8")
    return netlist
