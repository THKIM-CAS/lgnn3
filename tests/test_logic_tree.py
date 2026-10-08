import unittest

import torch

from light_dlgn.model import LogicTree


def _set_or_gates(tree: LogicTree) -> None:
    with torch.no_grad():
        for level in tree.logic_levels:
            level.logits.copy_(torch.tensor([-10.0, 10.0, 10.0, 10.0]).repeat(level.out_features, 1))


class LogicTreeTest(unittest.TestCase):
    def test_reachable_widths_for_five_inputs(self) -> None:
        x = torch.zeros(2, 5)
        self.assertEqual(LogicTree(5, 3)(x).shape, (2, 3))
        self.assertEqual(LogicTree(5, 2)(x).shape, (2, 2))
        self.assertEqual(LogicTree(5, 1)(x).shape, (2, 1))

    def test_bye_advances_unchanged(self) -> None:
        tree = LogicTree(5, 3, estimator="sigmoid", residual_init=False)
        _set_or_gates(tree)
        x = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.75]])
        torch.testing.assert_close(tree(x, discrete=True), torch.tensor([[1.0, 1.0, 0.75]]))

    def test_identity_tree(self) -> None:
        tree = LogicTree(5, 5)
        x = torch.randn(3, 5)
        self.assertEqual(len(tree.logic_levels), 0)
        torch.testing.assert_close(tree(x), x)

    def test_single_input_tree_is_identity(self) -> None:
        tree = LogicTree(1, 1)
        x = torch.tensor([[0.25], [0.75]])
        self.assertEqual(len(tree.logic_levels), 0)
        torch.testing.assert_close(tree(x), x)

    def test_continuous_tree_is_differentiable(self) -> None:
        tree = LogicTree(5, 1, estimator="sigmoid", residual_init=False)
        x = torch.rand(2, 5, requires_grad=True)
        tree(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(level.logits.grad is not None for level in tree.logic_levels))

    def test_invalid_width_and_input_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "not reachable"):
            LogicTree(5, 4)
        with self.assertRaisesRegex(ValueError, "expected x"):
            LogicTree(5, 1)(torch.zeros(2, 4))


if __name__ == "__main__":
    unittest.main()
