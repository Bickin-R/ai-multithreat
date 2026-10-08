"""Local synthetic tests for the COCO person dataset importer."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ai.training.datasets.import_person_dataset import import_dataset, validate_import
from ai.training.datasets.formats import PERSON_CLASSES, validate_dataset_splits
from ai.training.datasets.audit import audit_dataset


def fixture_image(seed, size=(64, 48)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


class PersonDatasetImporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="person-import-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.images = self.source / "images"
        self.images.mkdir(parents=True)
        self.output = self.root / "online-output"

    def tearDown(self):
        self.temp.cleanup()

    def save_coco(self, filename, rows, annotations=None, categories=None, licenses=None):
        path = self.source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"images": rows,
                    "categories": categories or [{"id": 1, "name": "person"},
                                                 {"id": 3, "name": "car"}],
                    "annotations": annotations or [], "licenses": licenses or []}
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def import_combined(self, annotation_path, **kwargs):
        kwargs.setdefault("download_date", "2026-10-08")
        return import_dataset(self.source, self.output, "fixture-set",
                              "https://example.invalid/dataset", "CC BY 4.0", "CC BY 4.0",
                              combined_annotations=annotation_path, **kwargs)

    def test_person_only_conversion_multiple_people_and_provenance(self):
        image_path = self.images / "people.png"
        fixture_image(3).save(image_path)
        annotation = self.save_coco("annotations/all.json",
            [{"id": 101, "file_name": "people.png", "width": 64, "height": 48}],
            [{"id": 1, "image_id": 101, "category_id": 1, "bbox": [2, 3, 10, 12]},
             {"id": 2, "image_id": 101, "category_id": 1, "bbox": [20, 10, 15, 30]},
             {"id": 3, "image_id": 101, "category_id": 3, "bbox": [1, 1, 5, 5]}])

        result = self.import_combined(annotation, source_version="v1")
        self.assertEqual(result["images_imported"], 1)
        self.assertEqual(result["person_annotations"], 2)
        self.assertEqual(result["non_person_annotations_discarded"], 1)
        row = json.loads((self.output / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["objects"], [
            {"class": 0, "bbox": [2.0, 3.0, 12.0, 15.0]},
            {"class": 0, "bbox": [20.0, 10.0, 35.0, 40.0]},
        ])
        self.assertFalse(row["reviewed"])
        self.assertEqual(row["source_id"], "101")
        self.assertEqual(row["source_group"], "fixture-set:train:image-101")
        self.assertEqual(row["license"], "CC BY 4.0")
        self.assertEqual(row["annotation_license"], "CC BY 4.0")
        provenance = json.loads((self.output / "source_manifest.json").read_text())
        self.assertEqual(provenance["license"], "CC BY 4.0")
        self.assertEqual(provenance["annotation_format"], "coco")
        self.assertEqual(validate_import(self.output), {"images": 1, "person_annotations": 2})
        copied = self.output / row["image"]
        self.assertTrue(copied.is_file())
        self.assertEqual(copied.read_bytes(), image_path.read_bytes())

    def test_combined_dataset_group_split_is_deterministic_and_nonleaking(self):
        rows = []
        annotations = []
        for index in range(20):
            filename = f"frame_{index:02}.png"
            fixture_image(index + 100).save(self.images / filename)
            rows.append({"id": index + 1, "file_name": filename,
                         "video_id": index // 2, "width": 64, "height": 48})
            annotations.append({"id": index + 1, "image_id": index + 1,
                                "category_id": 1, "bbox": [1, 1, 10, 10]})
        annotation = self.save_coco("annotations/all.json", rows, annotations)
        summary = self.import_combined(annotation)
        counts = summary["split_counts"]
        self.assertEqual(sum(counts.values()), 20)
        self.assertGreater(counts["train"], counts["val"])
        self.assertGreater(counts["train"], counts["test"])
        groups = {}
        for split in ("train", "val", "test"):
            for line in (self.output / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                source_id = int(row["source_id"]) - 1
                video = source_id // 2
                groups.setdefault(video, set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(validate_dataset_splits(
            self.output, self.output / "train.jsonl", self.output / "val.jsonl",
            self.output / "test.jsonl", "detection", PERSON_CLASSES),
            {"train": counts["train"], "validation": counts["val"], "test": counts["test"]})

    def test_official_split_annotations_are_preserved(self):
        split_annotations = {}
        for split, image_id in (("train", 1), ("val", 2), ("test", 3)):
            filename = f"{split}.png"
            fixture_image(image_id + 250).save(self.images / filename)
            split_annotations[split] = self.save_coco(
                f"annotations/{split}.json",
                [{"id": image_id, "file_name": filename, "width": 64, "height": 48}],
                [{"id": image_id, "image_id": image_id, "category_id": 1,
                  "bbox": [0, 0, 8, 8]}])
        result = import_dataset(self.source, self.output, "official-set",
                                "https://example.invalid/source", "License supplied by user",
                                "Annotation license supplied by user",
                                split_annotations=split_annotations,
                                download_date="2026-10-08")
        self.assertEqual(result["split_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(len((self.output / "train.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((self.output / "val.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((self.output / "test.jsonl").read_text().splitlines()), 1)

    def test_exact_and_near_duplicates_are_reported_without_overwrite(self):
        existing_dir = self.output / "images"
        existing_dir.mkdir(parents=True)
        base = fixture_image(45)
        base_path = existing_dir / "online_000001.png"
        base.save(base_path)
        original_bytes = base_path.read_bytes()
        near = base.copy()
        near.putpixel((0, 0), (near.getpixel((0, 0))[0] ^ 1,
                               near.getpixel((0, 0))[1], near.getpixel((0, 0))[2]))
        (self.images / "exact.png").write_bytes(original_bytes)
        near.save(self.images / "near.png")
        annotation = self.save_coco("annotations/all.json",
            [{"id": 1, "file_name": "exact.png"}, {"id": 2, "file_name": "near.png"}], [])

        result = self.import_combined(annotation)
        self.assertEqual(result["exact_duplicates"], 1)
        self.assertEqual(result["near_duplicates"], 1)
        self.assertEqual(result["images_imported"], 0)
        self.assertEqual(base_path.read_bytes(), original_bytes)
        self.assertEqual(len(result["skipped_files"]), 2)

    def test_unique_output_name_does_not_overwrite_existing_image(self):
        existing_dir = self.output / "images"
        existing_dir.mkdir(parents=True)
        existing = existing_dir / "online_000001.png"
        fixture_image(5).save(existing)
        preserved = existing.read_bytes()
        fixture_image(6).save(self.images / "new.png")
        annotation = self.save_coco("annotations/all.json",
            [{"id": 1, "file_name": "new.png"}], [])
        result = self.import_combined(annotation)
        row = json.loads((self.output / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["image"], "images/online_000002.png")
        self.assertEqual(existing.read_bytes(), preserved)
        self.assertEqual(result["images_imported"], 1)

    def test_source_image_license_and_attribution_are_preserved(self):
        fixture_image(29).save(self.images / "licensed.png")
        annotation = self.save_coco("annotations/all.json",
            [{"id": 29, "file_name": "licensed.png", "license": 7,
              "flickr_url": "https://images.example.invalid/29"}], [],
            licenses=[{"id": 7, "name": "CC BY 2.0", "url": "https://creativecommons.org/licenses/by/2.0/"}])
        self.import_combined(annotation)
        row = json.loads((self.output / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["license"], "CC BY 2.0")
        self.assertEqual(row["source_license_url"], "https://creativecommons.org/licenses/by/2.0/")
        self.assertEqual(row["source_license_id"], 7)
        self.assertEqual(row["source_image_url"], "https://images.example.invalid/29")
        provenance = json.loads((self.output / "source_manifest.json").read_text())
        self.assertEqual(provenance["image_licenses_used"], ["CC BY 2.0"])

    def test_unresolved_image_license_id_is_rejected(self):
        fixture_image(30).save(self.images / "unlicensed-id.png")
        annotation = self.save_coco("annotations/all.json",
            [{"id": 30, "file_name": "unlicensed-id.png", "license": 42}], [])
        with self.assertRaisesRegex(ValueError, "unknown COCO license ID"):
            self.import_combined(annotation)

    def test_corrupted_images_invalid_boxes_license_and_format_are_rejected(self):
        (self.images / "broken.jpg").write_bytes(b"not an image")
        annotation = self.save_coco("annotations/all.json",
            [{"id": 1, "file_name": "broken.jpg"}], [])
        with self.assertRaisesRegex(ValueError, "Corrupted or unreadable"):
            self.import_combined(annotation)
        self.assertFalse(self.output.exists())

        fixture_image(7).save(self.images / "badbox.png")
        badbox = self.save_coco("annotations/bad.json",
            [{"id": 2, "file_name": "badbox.png"}],
            [{"id": 1, "image_id": 2, "category_id": 1, "bbox": [60, 0, 10, 10]}])
        with self.assertRaisesRegex(ValueError, "outside image bounds"):
            self.import_combined(badbox)

        with self.assertRaisesRegex(ValueError, "verified source license"):
            import_dataset(self.source, self.output, "fixture", "https://example.invalid", "unknown", "CC BY 4.0",
                           combined_annotations=badbox)
        with self.assertRaisesRegex(ValueError, "Unsupported annotation format"):
            import_dataset(self.source, self.output, "fixture", "https://example.invalid", "CC BY", "CC BY",
                           combined_annotations=badbox, annotation_format="voc")

    def test_missing_person_class_and_malformed_provenance_are_rejected(self):
        fixture_image(8).save(self.images / "car.png")
        no_person = self.save_coco("annotations/car.json",
            [{"id": 1, "file_name": "car.png"}], [], categories=[{"id": 3, "name": "car"}])
        with self.assertRaisesRegex(ValueError, "No source category named 'person'"):
            self.import_combined(no_person)

        self.output.mkdir()
        (self.output / "source_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Provenance missing required fields"):
            validate_import(self.output)

    def test_dataset_validators_detect_removed_source_provenance(self):
        fixture_image(31).save(self.images / "provenance.png")
        annotation = self.save_coco("annotations/all.json",
            [{"id": 31, "file_name": "provenance.png"}], [])
        self.import_combined(annotation)
        (self.output / "source_manifest.json").unlink()
        report = audit_dataset(self.output, "person")
        self.assertTrue(any("missing provenance file" in error
                            for error in report["splits"]["train"]["errors"]))
        with self.assertRaisesRegex(ValueError, "missing provenance file"):
            validate_dataset_splits(self.output, self.output / "train.jsonl",
                                    self.output / "val.jsonl", self.output / "test.jsonl",
                                    "detection", PERSON_CLASSES)

if __name__ == "__main__":
    unittest.main()
