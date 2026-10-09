"""Small synthetic tests for prediction visualization overlays."""
import unittest
import json
import tempfile
from pathlib import Path

from PIL import Image
import torch

from ai.models import LegacyPersonDetector20, PersonDetectorModel
from ai.training.visualize_person_predictions import annotate_image, run_visualization


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

    def test_legacy20_checkpoint_runs_through_visualizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            Image.new("RGB", (64, 48), "gray").save(root / "images" / "sample.png")
            (root / "test.jsonl").write_text(json.dumps({
                "image": "images/sample.png",
                "objects": [{"class": 0, "bbox": [10, 5, 30, 40]}],
            }) + "\n", encoding="utf-8")
            checkpoint = root / "legacy.pt"
            torch.save({
                "format_version": 1,
                "model_name": "PersonDetectorModel",
                "state_dict": LegacyPersonDetector20().state_dict(),
                "class_names": ["person"],
                "input_size": [320, 320],
            }, checkpoint)
            output = root / "out"
            result = run_visualization(root, checkpoint, "cpu", 0.1, output, "legacy20")
            self.assertEqual(result["images_processed"], 1)
            self.assertEqual(result["total_ground_truth_boxes"], 1)
            self.assertTrue(next(output.glob("*.png")).is_file())

    def test_checkpoint_architecture_must_be_selected_correctly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            Image.new("RGB", (32, 32), "gray").save(root / "images" / "sample.png")
            (root / "test.jsonl").write_text(json.dumps({
                "image": "images/sample.png", "objects": [],
            }) + "\n", encoding="utf-8")
            checkpoint = root / "current.pt"
            model = PersonDetectorModel()
            torch.save({
                "format_version": 1,
                "model_name": model.__class__.__name__,
                "state_dict": model.state_dict(),
                "class_names": ["person"],
                "input_size": [320, 320],
                "architecture_id": model.architecture_id,
                "output_grid": list(model.output_grid),
            }, checkpoint)
            with self.assertRaisesRegex(ValueError, "legacy20 accepts only"):
                run_visualization(root, checkpoint, "cpu", 0.1, root / "wrong", "legacy20")


if __name__ == "__main__":
    unittest.main()
