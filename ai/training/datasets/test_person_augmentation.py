"""Synthetic tests for opt-in, box-safe Person training augmentation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw

from ai.training.datasets.formats import DetectionDataset, SIZE, load_rgb


class PersonAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="person-augmentation-")
        self.root = Path(self.temp.name)
        image = Image.new("RGB", (100, 50), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 49, 49), fill=(220, 20, 10))
        draw.rectangle((50, 0, 99, 49), fill=(10, 20, 220))
        image.save(self.root / "sample.png")
        self.row = {"image": "sample.png", "objects": [
            {"class": 0, "bbox": [10, 10, 30, 30]}]}

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, name):
        path = self.root / name
        path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        return path

    def test_horizontal_flip_mirrors_center_and_preserves_size(self):
        dataset = DetectionDataset(self.root, self.manifest("train.jsonl"), augment=True)
        with patch("ai.training.datasets.formats.random.random", return_value=0.1), \
             patch("ai.training.datasets.formats.random.uniform", return_value=1.0):
            image, boxes = dataset[0]
        self.assertEqual(tuple(image.shape), (3, *SIZE))
        self.assertAlmostEqual(float(boxes[0, 1]), 0.8)
        self.assertAlmostEqual(float(boxes[0, 2]), 0.4)
        self.assertAlmostEqual(float(boxes[0, 3]), 0.2)
        self.assertAlmostEqual(float(boxes[0, 4]), 0.4)

    def test_mild_color_augmentation_returns_valid_rgb_tensor_and_boxes(self):
        dataset = DetectionDataset(self.root, self.manifest("train.jsonl"), augment=True)
        with patch("ai.training.datasets.formats.random.random", return_value=0.9), \
             patch("ai.training.datasets.formats.random.uniform", side_effect=(1.08, 0.94)):
            image, boxes = dataset[0]
        self.assertEqual(tuple(image.shape), (3, *SIZE))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(torch.isfinite(image).all())
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertEqual(boxes.shape, (1, 5))
        self.assertTrue(torch.isfinite(boxes).all())
        self.assertTrue(torch.all((boxes[:, 1:3] >= 0) & (boxes[:, 1:3] <= 1)))
        self.assertTrue(torch.all((boxes[:, 3:5] > 0) & (boxes[:, 3:5] <= 1)))

    def test_validation_and_test_datasets_are_unaugmented_by_default(self):
        expected = load_rgb(self.root / "sample.png")
        for split in ("val", "test"):
            dataset = DetectionDataset(self.root, self.manifest(f"{split}.jsonl"))
            with patch("ai.training.datasets.formats.random.random",
                       side_effect=AssertionError("unaugmented dataset consumed randomness")), \
                 patch("ai.training.datasets.formats.random.uniform",
                       side_effect=AssertionError("unaugmented dataset consumed randomness")):
                first_image, first_boxes = dataset[0]
                second_image, second_boxes = dataset[0]
            self.assertFalse(dataset.augment)
            self.assertTrue(torch.equal(first_image, expected))
            self.assertTrue(torch.equal(first_image, second_image))
            self.assertTrue(torch.equal(first_boxes, second_boxes))
            self.assertAlmostEqual(float(first_boxes[0, 1]), 0.2)


if __name__ == "__main__":
    unittest.main()
