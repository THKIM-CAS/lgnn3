import unittest

from train import parse_widths


class ParseWidthsTest(unittest.TestCase):
    def test_legacy_integer_widths_remain_supported(self) -> None:
        self.assertEqual(parse_widths("100,100,100", (16_000,)), (100, 100, 100))

    def test_grouped_tree_widths_are_parsed(self) -> None:
        self.assertEqual(
            parse_widths("300,(10,2),100,100", (16_000,)),
            (300, (10, 2), 100, 100),
        )

    def test_invalid_width_syntax_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer or a"):
            parse_widths("300,(10,2,1),100", (16_000,))


if __name__ == "__main__":
    unittest.main()
