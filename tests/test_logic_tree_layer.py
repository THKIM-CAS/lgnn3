import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from light_dlgn.export_verilog import (
    extract_logic_netlist,
    load_model_from_checkpoint,
    netlist_to_verilog,
    verify_netlist,
)
from light_dlgn.model import InputWiseLogicLayer, LightDLGN, LogicTreeLayer


def _set_or_gates(layer: LogicTreeLayer) -> None:
    with torch.no_grad():
        for tree in layer.logic_trees:
            for gates in tree.logic_levels:
                gates.logits.copy_(torch.tensor([-10.0, 10.0, 10.0, 10.0]).repeat(gates.out_features, 1))


class LogicTreeLayerTest(unittest.TestCase):
    def test_contiguous_groups_are_concatenated_in_group_order(self) -> None:
        layer = LogicTreeLayer(6, 2, 2, estimator="sigmoid", residual_init=False)
        _set_or_gates(layer)
        x = torch.tensor([[0.0, 1.0, 0.2, 1.0, 0.0, 0.7]])
        torch.testing.assert_close(layer(x, discrete=True), torch.tensor([[1.0, 0.2, 1.0, 0.7]]))

    def test_invalid_grouping_and_tree_width_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            LogicTreeLayer(5, 2, 1)
        with self.assertRaisesRegex(ValueError, "not reachable"):
            LogicTreeLayer(10, 2, 4)

    def test_mixed_width_model_uses_tree_layer_and_output_width(self) -> None:
        model = LightDLGN(
            image_shape=(1, 1, 4),
            num_classes=2,
            widths=(8, (2, 1), 2),
            num_thresholds=1,
            tau=1.0,
            seed=0,
        )
        self.assertEqual(model.widths, (8, (2, 1), 2))
        self.assertIsInstance(model.logic_layers[0], InputWiseLogicLayer)
        self.assertIsInstance(model.logic_layers[1], LogicTreeLayer)
        self.assertIsInstance(model.logic_layers[2], InputWiseLogicLayer)
        self.assertEqual(model.logic_layers[1].out_features, 2)
        self.assertEqual(model(torch.rand(3, 1, 1, 4), discrete=False).shape, (3, 2))

    def test_legacy_integer_widths_use_only_input_wise_layers(self) -> None:
        model = LightDLGN(
            image_shape=(1, 1, 4),
            num_classes=2,
            widths=(8, 4, 2),
            num_thresholds=1,
            tau=1.0,
            seed=0,
        )
        self.assertTrue(all(isinstance(layer, InputWiseLogicLayer) for layer in model.logic_layers))
        self.assertEqual(
            set(model.state_dict()),
            {
                "thresholds",
                "logic_layers.0.logits",
                "logic_layers.0.left_indices",
                "logic_layers.0.right_indices",
                "logic_layers.1.logits",
                "logic_layers.1.left_indices",
                "logic_layers.1.right_indices",
                "logic_layers.2.logits",
                "logic_layers.2.left_indices",
                "logic_layers.2.right_indices",
            },
        )

    def test_nested_width_checkpoint_round_trip(self) -> None:
        model = LightDLGN(
            image_shape=(1, 1, 4),
            num_classes=2,
            widths=(8, (2, 1), 2),
            num_thresholds=1,
            tau=1.0,
            estimator="sigmoid",
            residual_init=False,
            seed=7,
        )
        model_config = {
            "image_shape": model.image_shape,
            "num_classes": model.num_classes,
            "widths": model.widths,
            "num_thresholds": model.num_thresholds,
            "tau": model.tau,
            "estimator": model.estimator,
            "residual_init": model.residual_init,
            "seed": 7,
        }
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            torch.save({"model_config": model_config, "model_state": model.state_dict()}, checkpoint_path)
            loaded = load_model_from_checkpoint(checkpoint_path)
        self.assertEqual(loaded.widths, model.widths)
        self.assertIsInstance(loaded.logic_layers[1], LogicTreeLayer)

    def test_tree_models_export_to_verified_tournament_netlists(self) -> None:
        model = LightDLGN(
            image_shape=(1, 1, 4),
            num_classes=2,
            widths=(6, (2, 2), 2),
            num_thresholds=1,
            tau=1.0,
            seed=0,
        )
        netlist = extract_logic_netlist(model)
        self.assertEqual(netlist.layer_widths, (6, 4, 2))
        self.assertEqual(netlist.output_width, 2)
        for bye_gate in (netlist.gates_by_layer[1][1], netlist.gates_by_layer[1][3]):
            self.assertEqual(bye_gate.truth_table, (0, 0, 1, 1))
            self.assertEqual(bye_gate.left_index, bye_gate.right_index)
        verify_netlist(model, netlist, samples=32, seed=3)
        self.assertIn("module tree_model(", netlist_to_verilog(netlist, module_name="tree_model"))


if __name__ == "__main__":
    unittest.main()
