"""Integration contract tests between the Person importer and model workflow."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
import torch

from ai.training.cli import main as training_main
from ai.training.datasets.audit import audit_dataset
from ai.training.datasets.formats import DetectionDataset, validate_dataset_splits
from ai.training.datasets.source_provenance import MANIFEST_FIELDS
from ai.models import LegacyPersonDetector20
from ai.training.evaluate_person import run_evaluation


class PersonDatasetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="person-dataset-integration-")
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        manifest = {
            "dataset_name": "fixture",
            "dataset_url": "https://example.invalid/fixture",
            "license": "CC BY 4.0",
            "annotation_license": "CC BY 4.0",
            "download_date": "2026-10-01",
            "images_imported": 3,
            "person_annotations": 3,
            "annotation_format": "coco",
            "image_licenses_used": ["CC BY 4.0"],
        }
        self.assertEqual(set(manifest), set(MANIFEST_FIELDS))
        (self.root / "source_manifest.json").write_text(json.dumps(manifest))
        for index, split in enumerate(("train", "val", "test")):
            Image.new("RGB", (80, 60), (20 + index * 30, 30, 40)).save(
                self.root / "images" / f"{split}.jpg")
            row = {
                "image": f"images/{split}.jpg",
                "objects": [{"class": 0, "bbox": [10, 5, 40, 50]}],
                "reviewed": True,
                "source_name": "fixture",
                "source_url": "https://example.invalid/fixture",
                "source_id": split,
                "license": "CC BY 4.0",
                "annotation_license": "CC BY 4.0",
                "source_group": f"fixture:{split}:{split}-group",
            }
            (self.root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_person1b_manifest_contract_and_split_reading(self):
        counts = validate_dataset_splits(
            self.root, self.root / "train.jsonl", self.root / "val.jsonl",
            self.root / "test.jsonl", "detection", ["person"])
        self.assertEqual(counts, {"train": 1, "validation": 1, "test": 1})
        train = DetectionDataset(self.root, self.root / "train.jsonl", ["person"], augment=True)
        val = DetectionDataset(self.root, self.root / "val.jsonl", ["person"], augment=False)
        test = DetectionDataset(self.root, self.root / "test.jsonl", ["person"], augment=False)
        self.assertEqual([row["image"] for row in train.rows], ["images/train.jpg"])
        self.assertEqual([row["image"] for row in val.rows], ["images/val.jpg"])
        self.assertEqual([row["image"] for row in test.rows], ["images/test.jpg"])
        _, boxes = test[0]
        # Pixel XYXY [10,5,40,50] on 80x60 becomes normalized CXCYWH.
        for actual, expected in zip(boxes[0].tolist(),
                                    [0.0, 0.3125, 0.4583333, 0.375, 0.75]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_training_entrypoint_uses_only_train_and_validation_manifests(self):
        captured = {}

        def capture(model, train_ds, val_ds, *args, **kwargs):
            captured["train"] = [row["image"] for row in train_ds.rows]
            captured["val"] = [row["image"] for row in val_ds.rows]
            captured["train_augment"] = train_ds.augment
            captured["val_augment"] = val_ds.augment

        args = ["train_person.py", "--data", str(self.root), "--device", "cpu",
                "--checkpoint", str(self.root / "unused.pt")]
        with mock.patch.object(sys, "argv", args), \
             mock.patch("ai.training.cli.train_detector", side_effect=capture):
            training_main("person")
        self.assertEqual(captured["train"], ["images/train.jpg"])
        self.assertEqual(captured["val"], ["images/val.jpg"])
        self.assertNotIn("images/test.jpg", captured["train"] + captured["val"])
        self.assertTrue(captured["train_augment"])
        self.assertFalse(captured["val_augment"])

    def test_imported_unreviewed_labels_are_reported_and_blocked(self):
        row = json.loads((self.root / "train.jsonl").read_text())
        row["reviewed"] = False
        (self.root / "train.jsonl").write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "reviewed=true"):
            DetectionDataset(self.root, self.root / "train.jsonl", ["person"])
        report = audit_dataset(self.root, "person")
        self.assertEqual(report["splits"]["train"]["invalid_samples"], 1)
        self.assertTrue(any("reviewed=true" in error for error in report["splits"]["train"]["errors"]))

    def test_final_evaluator_loads_only_test_manifest_with_legacy20_checkpoint(self):
        # This root intentionally has no train/val manifests or images: final
        # evaluation must only ask the dataset loader for test.jsonl.
        test_root = self.root / "heldout"
        (test_root / "images").mkdir(parents=True)
        Image.new("RGB", (80, 60), (90, 30, 40)).save(test_root / "images/test.jpg")
        (test_root / "test.jsonl").write_text(json.dumps({
            "image": "images/test.jpg",
            "objects": [{"class": 0, "bbox": [10, 5, 40, 50]}],
        }) + "\n")
        checkpoint = test_root / "legacy20.pt"
        model = LegacyPersonDetector20()
        torch.save({
            "format_version": 1,
            "model_name": "PersonDetectorModel",
            "state_dict": model.state_dict(),
            "class_names": ["person"],
            "input_size": [320, 320],
        }, checkpoint)
        report = run_evaluation(test_root, checkpoint, "cpu", 0.1, "legacy20")
        self.assertEqual(report["images"], 1)
        self.assertEqual(report["ground_truth"], 1)


if __name__ == "__main__":
    unittest.main()
