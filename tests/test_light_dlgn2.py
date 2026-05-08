from __future__ import annotations

import unittest

import torch

from light_dlgn import LightDLGN2


def _small_model(**overrides: object) -> LightDLGN2:
    kwargs = {
        "image_shape": (1, 2, 2),
        "num_classes": 3,
        "steps_per_class": 2,
        "population": 4,
        "feedback_features": 2,
        "step_widths": (6,),
        "num_thresholds": 1,
        "tau": 2.0,
        "seed": 0,
    }
    kwargs.update(overrides)
    return LightDLGN2(**kwargs)


class LightDLGN2Test(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = _small_model()
        x = torch.rand(5, 1, 2, 2)

        logits = model(x, discrete=False)

        self.assertEqual(tuple(logits.shape), (5, 3))

    def test_requires_uniform_input_partitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by steps_per_class"):
            _small_model(image_shape=(1, 1, 3), steps_per_class=2)

    def test_requires_final_step_width_to_match_feedback_and_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "feedback_features \\+ population"):
            _small_model(step_widths=(5,))

    def test_default_discrete_mode_follows_training_state(self) -> None:
        model = _small_model()
        x = torch.rand(2, 1, 2, 2)

        model.train()
        train_default = model(x)
        train_continuous = model(x, discrete=False)

        model.eval()
        eval_default = model(x)
        eval_discrete = model(x, discrete=True)

        self.assertTrue(torch.allclose(train_default, train_continuous))
        self.assertTrue(torch.allclose(eval_default, eval_discrete))

    def test_seed_deterministically_initializes_model(self) -> None:
        model_a = _small_model(seed=123)
        model_b = _small_model(seed=123)
        x = torch.rand(3, 1, 2, 2)

        for key, value in model_a.state_dict().items():
            self.assertTrue(torch.equal(value, model_b.state_dict()[key]), key)
        self.assertTrue(torch.allclose(model_a(x), model_b(x)))

    def test_zero_feedback_features_runs(self) -> None:
        model = _small_model(feedback_features=0, population=2, step_widths=(2,))
        x = torch.rand(4, 1, 2, 2)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_class_step_layers_are_independent(self) -> None:
        model = LightDLGN2(
            image_shape=(1, 1, 1),
            num_classes=2,
            steps_per_class=1,
            population=1,
            feedback_features=0,
            step_widths=(1,),
            num_thresholds=1,
            tau=1.0,
            estimator="sigmoid",
            seed=0,
        )
        with torch.no_grad():
            model.step_layers[0][0].logits.fill_(-10.0)
            model.step_layers[1][0].logits.fill_(10.0)

        x = torch.ones(1, 1, 1, 1)

        logits = model(x, discrete=True)

        self.assertTrue(torch.equal(logits, torch.tensor([[0.0, 1.0]])))

    def test_feedback_resets_for_each_class(self) -> None:
        model = LightDLGN2(
            image_shape=(1, 1, 1),
            num_classes=2,
            steps_per_class=1,
            population=1,
            feedback_features=1,
            step_widths=(2,),
            num_thresholds=1,
            tau=1.0,
            estimator="sigmoid",
            seed=0,
        )
        with torch.no_grad():
            for class_layers in model.step_layers:
                layer = class_layers[0]
                layer.left_indices.copy_(torch.tensor([0, 0]))
                layer.right_indices.copy_(torch.tensor([1, 1]))
                layer.logits.copy_(
                    torch.tensor(
                        [
                            [-10.0, -10.0, 10.0, 10.0],
                            [-10.0, 10.0, -10.0, 10.0],
                        ]
                    )
                )

        x = torch.ones(1, 1, 1, 1)

        logits = model(x, discrete=True)

        self.assertTrue(torch.equal(logits, torch.zeros(1, 2)))


if __name__ == "__main__":
    unittest.main()
