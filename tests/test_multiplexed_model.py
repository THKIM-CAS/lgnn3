from __future__ import annotations

import unittest

import torch

from light_dlgn.model import MultiplexedLightDLGN


class MultiplexedLightDLGNTests(unittest.TestCase):
    def test_layer_input_widths_include_class_code_every_time(self) -> None:
        model = MultiplexedLightDLGN(
            image_shape=(1, 2, 2),
            num_classes=3,
            widths=(5, 7, 11, 13),
            num_thresholds=2,
            tau=1.0,
            seed=0,
        )

        encoded_dim = 1 * 2 * 2 * 2
        self.assertEqual(model.num_logic_layers, 3)
        self.assertEqual(model.logic_layers[0].in_features, encoded_dim + 5)
        self.assertEqual(model.logic_layers[1].in_features, 7 + 5)
        self.assertEqual(model.logic_layers[2].in_features, 11 + 5)

    def test_class_code_logits_has_per_layer_shape(self) -> None:
        model = MultiplexedLightDLGN(
            image_shape=(1, 2, 2),
            num_classes=4,
            widths=(6, 8, 10),
            num_thresholds=1,
            tau=1.0,
            seed=0,
        )

        self.assertEqual(tuple(model.class_code_logits.shape), (2, 4, 6))
        self.assertEqual(tuple(model.class_codes(discrete=False).shape), (2, 4, 6))

    def test_forward_shape_in_continuous_and_discrete_modes(self) -> None:
        model = MultiplexedLightDLGN(
            image_shape=(1, 2, 2),
            num_classes=3,
            widths=(4, 6, 9),
            num_thresholds=2,
            tau=1.0,
            seed=0,
        )
        x = torch.rand(5, 1, 2, 2)

        continuous_logits = model(x, discrete=False)
        discrete_logits = model(x, discrete=True)

        self.assertEqual(tuple(continuous_logits.shape), (5, 3))
        self.assertEqual(tuple(discrete_logits.shape), (5, 3))

    def test_discrete_class_codes_are_binary(self) -> None:
        model = MultiplexedLightDLGN(
            image_shape=(1, 2, 2),
            num_classes=3,
            widths=(4, 6, 9),
            num_thresholds=1,
            tau=1.0,
            seed=0,
        )

        codes = model.class_codes(discrete=True)
        self.assertTrue(torch.all((codes == 0) | (codes == 1)).item())

    def test_legacy_checkpoint_shape_raises_clear_error(self) -> None:
        model = MultiplexedLightDLGN(
            image_shape=(1, 2, 2),
            num_classes=3,
            widths=(4, 6, 9),
            num_thresholds=1,
            tau=1.0,
            seed=0,
        )
        state_dict = model.state_dict()
        state_dict["class_code_logits"] = torch.zeros(3, 4)

        with self.assertRaisesRegex(RuntimeError, "legacy multiplexed checkpoints are incompatible"):
            model.load_state_dict(state_dict)


if __name__ == "__main__":
    unittest.main()
