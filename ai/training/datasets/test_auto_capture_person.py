"""Webcam-free tests for automatic person image collection."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from ai.training.datasets import auto_capture_person as auto
from ai.training.datasets.annotate_person import IMAGE_SUFFIXES, person_record


def detailed_frame(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(35, 220, size=(240, 320, 3), dtype=np.uint8)


class FakeCamera:
    def __init__(self, index, frame=None, readable=True):
        self.index = index
        self.frame = detailed_frame() if frame is None else frame
        self.readable = readable
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return (self.readable, self.frame.copy() if self.readable else None)

    def release(self):
        self.released = True


class AutoCapturePersonTests(unittest.TestCase):
    def test_camera_diagnostic_reports_runtime_and_frame_read(self):
        camera = FakeCamera(0)
        output = io.StringIO()
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(auto.cv2, "VideoCapture", return_value=camera), \
             redirect_stdout(output):
            result = auto.diagnose_camera(0)
        self.assertEqual(result, 0)
        self.assertTrue(camera.released)
        report = output.getvalue()
        self.assertIn("Python executable:", report)
        self.assertIn(f"OpenCV version: {cv2.__version__}", report)
        self.assertIn("Requested camera index: 0", report)
        self.assertIn("/dev/video0 exists: True", report)
        self.assertIn("OpenCV can open camera: True", report)
        self.assertIn("OpenCV can read a frame: True", report)

    def test_unique_names_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            existing = output / "smoke_000001.jpg"
            existing.write_bytes(b"keep this existing file")
            first = auto._save_without_overwrite(detailed_frame(), output, "smoke")
            second = auto._save_without_overwrite(detailed_frame(2), output, "smoke")
            self.assertEqual(first.name, "smoke_000002.jpg")
            self.assertEqual(second.name, "smoke_000003.jpg")
            self.assertEqual(existing.read_bytes(), b"keep this existing file")

    def test_quality_filters_resolution_blur_exposure_and_near_duplicates(self):
        small = np.zeros((200, 300, 3), dtype=np.uint8)
        self.assertEqual(auto.frame_quality(small)[1], "resolution")

        blurry = np.full((240, 320, 3), 125, dtype=np.uint8)
        self.assertEqual(auto.frame_quality(blurry)[1], "blur")

        rng = np.random.default_rng(2)
        overexposed = rng.integers(245, 256, size=(240, 320, 3), dtype=np.uint8)
        underexposed = rng.integers(0, 10, size=(240, 320, 3), dtype=np.uint8)
        self.assertEqual(auto.frame_quality(overexposed)[1], "exposure")
        self.assertEqual(auto.frame_quality(underexposed)[1], "exposure")

        frame = detailed_frame()
        self.assertEqual(auto.frame_quality(frame, frame.copy())[1], "duplicate")

    def test_metadata_jsonl_and_annotation_format_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            image_dir = Path(temp) / "images"
            camera = FakeCamera(0)
            with mock.patch.object(auto.cv2, "VideoCapture", return_value=camera), \
                 mock.patch.object(auto.cv2, "namedWindow"), \
                 mock.patch.object(auto.cv2, "imshow"), \
                 mock.patch.object(auto.cv2, "waitKey", return_value=255), \
                 mock.patch.object(auto.cv2, "destroyAllWindows"):
                result = auto.capture(0, "unit", 1, 1.0, image_dir)

            self.assertEqual(result, 0)
            self.assertTrue(camera.released)
            files = list(image_dir.glob("unit_*.jpg"))
            self.assertEqual(len(files), 1)
            metadata_path = image_dir / "capture_metadata.jsonl"
            rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "webcam")
            self.assertEqual(rows[0]["session"], "unit")
            self.assertEqual(rows[0]["image"], files[0].name)

            self.assertIn(files[0].suffix.lower(), IMAGE_SUFFIXES)
            with Image.open(files[0]) as image:
                width, height = image.size
            record = person_record(f"images/{files[0].name}", [], width, height)
            self.assertEqual(record, {"image": f"images/{files[0].name}", "objects": []})

    def test_q_and_escape_stop_and_release_camera(self):
        for key in (ord("q"), 27):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                camera = FakeCamera(0)
                with mock.patch.object(auto.cv2, "VideoCapture", return_value=camera), \
                     mock.patch.object(auto.cv2, "namedWindow"), \
                     mock.patch.object(auto.cv2, "imshow"), \
                     mock.patch.object(auto.cv2, "waitKey", return_value=key), \
                     mock.patch.object(auto.cv2, "destroyAllWindows"):
                    result = auto.capture(0, "quit", 10, 0.000001, temp)
                self.assertEqual(result, 0)
                self.assertTrue(camera.released)
                self.assertEqual(len(list(Path(temp).glob("quit_*.jpg"))), 1)

    def test_read_failure_releases_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            camera = FakeCamera(0, readable=False)
            with mock.patch.object(auto.cv2, "VideoCapture", return_value=camera), \
                 mock.patch.object(auto.cv2, "namedWindow"), \
                 mock.patch.object(auto.cv2, "destroyAllWindows"):
                result = auto.capture(0, "failed", 5, 1.0, temp)
            self.assertEqual(result, 2)
            self.assertTrue(camera.released)

    def test_cli_defaults_and_arguments(self):
        args = auto._build_parser().parse_args([])
        self.assertEqual((args.camera, args.count, args.interval, args.output),
                         (0, 500, 1.0, "data/person/images"))
        args = auto._build_parser().parse_args([
            "--camera", "2", "--session", "manual", "--count", "7",
            "--interval", "0.5", "--output", "scratch/images",
        ])
        self.assertEqual((args.camera, args.session, args.count, args.interval, args.output),
                         (2, "manual", 7, 0.5, "scratch/images"))
        args = auto._build_parser().parse_args(["--check-camera", "--camera", "2"])
        self.assertTrue(args.check_camera)
        self.assertEqual(args.camera, 2)
        with mock.patch.object(auto, "capture", return_value=0) as capture:
            self.assertEqual(auto.main([]), 0)
        capture.assert_called_once_with(0, None, 500, 1.0, "data/person/images")
        with mock.patch.object(auto, "diagnose_camera", return_value=0) as diagnose:
            self.assertEqual(auto.main(["--check-camera", "--camera", "2"]), 0)
        diagnose.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
