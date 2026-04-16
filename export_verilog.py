from __future__ import annotations

import argparse
from pathlib import Path

from light_dlgn.export_verilog import write_verilog_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained discrete Light DLGN as a combinational Verilog-2005 module."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-name", type=str, default="light_dlgn_logic")
    parser.add_argument("--verify-samples", type=int, default=16)
    parser.add_argument("--verify-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    netlist = write_verilog_module(
        args.checkpoint,
        args.output,
        module_name=args.module_name,
        verify_samples=args.verify_samples,
        verify_seed=args.verify_seed,
    )
    print(
        f"wrote {args.output} "
        f"(inputs={netlist.encoded_dim}, "
        f"layers={','.join(str(width) for width in netlist.layer_widths)}, "
        f"classes={netlist.num_classes}, "
        f"bits_per_class={netlist.group_size})"
    )


if __name__ == "__main__":
    main()
