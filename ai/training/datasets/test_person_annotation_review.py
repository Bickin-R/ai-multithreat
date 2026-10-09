"""Tests for non-destructive Person annotation review records."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from ai.training.datasets.review_person_annotations import (
    FLAG_NAMES, build_summary, draw_annotation_overlay, load_reviews,
    render_summary, save_review,
)


class PersonAnnotationReviewTests(unittest.TestCase):
    def test_overlay_draws_existing_boxes_without_mutating_image(self):
        source = Image.new("RGB", (32, 32), "black")
        result = draw_annotation_overlay(source, [{"bbox": [4, 5, 20, 25]}])
        self.assertEqual(source.getpixel((4, 5)), (0, 0, 0))
        self.assertEqual(result.getpixel((4, 5)), (40, 255, 80))
        self.assertIsNot(source, result)

    def test_review_is_separate_and_manifest_bytes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = json.dumps({"image": "images/a.jpg", "objects": [
                {"class": 0, "bbox": [1, 2, 20, 30]},
            ]}) + "\n"
            manifest = root / "train.jsonl"
            manifest.write_text(original, encoding="utf-8")
            review_path = root / "reviews" / "results.jsonl"
            flags = {name: name == "incorrect_boxes" for name in FLAG_NAMES}

            result = save_review(review_path, "train", "images/a.jpg", True,
                                 flags, "Check the lower edge")

            self.assertEqual(manifest.read_text(encoding="utf-8"), original)
            self.assertTrue(result["reviewed_at"].endswith("+00:00"))
            self.assertTrue(review_path.is_file())
            self.assertTrue(load_reviews(review_path)[("train", "images/a.jpg")]["flags"]["incorrect_boxes"])

    def test_summary_lists_pending_reviewed_and_flagged_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {"image": f"images/{name}.jpg", "objects": [{"class": 0, "bbox": [1, 1, 10, 10]}]}
                for name in ("reviewed", "flagged", "pending")
            ]
            (root / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            review_path = root / "review_results.jsonl"
            save_review(review_path, "train", "images/reviewed.jpg", True,
                        {name: False for name in FLAG_NAMES})
            save_review(review_path, "train", "images/flagged.jpg", True,
                        {name: name == "missing_people" for name in FLAG_NAMES}, "Possible occluded person")

            summary = build_summary(root, "train", review_path)
            rendered = render_summary(summary)

            self.assertEqual((summary["total"], summary["reviewed_count"],
                              summary["pending_count"], summary["flagged_count"]), (3, 2, 1, 1))
            self.assertEqual(summary["pending"][0]["image"], "images/pending.jpg")
            self.assertIn("images/flagged.jpg", rendered)
            self.assertIn("images/pending.jpg", rendered)


if __name__ == "__main__":
    unittest.main()
