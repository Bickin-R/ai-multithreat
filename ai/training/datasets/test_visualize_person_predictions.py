"""Small synthetic tests for prediction visualization overlays."""
import unittest

from PIL import Image

from ai.training.visualize_person_predictions import annotate_image


class PersonPredictionVisualizationTests(unittest.TestCase):
    def test_annotation_draws_gt_and_prediction_without_mutating_source(self):
        source = Image.new("RGB", (40, 40), "black")
        annotated = annotate_image(
            source,
            [[2, 2, 16, 16]],
            [{"bbox": [22, 22, 36, 36], "confidence": 0.73}],
        )
        self.assertEqual(source.getpixel((2, 2)), (0, 0, 0))
        self.assertEqual(annotated.getpixel((2, 2)), (0, 255, 0))
        self.assertEqual(annotated.getpixel((22, 22)), (255, 0, 0))
        self.assertIsNot(annotated, source)


if __name__ == "__main__":
    unittest.main()
