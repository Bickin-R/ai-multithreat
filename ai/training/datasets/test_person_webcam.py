"""Tests for live Person webcam preprocessing and explicit architecture choice."""
import unittest
from unittest import mock

import numpy as np
import torch

from ai.inference.common import load_frame
from ai.inference.person_webcam import build_parser, build_inference, annotate_frame
from ai.models import LegacyPersonDetector20


class PersonWebcamTests(unittest.TestCase):
    def test_shared_frame_preprocessing_returns_rgb_model_tensor(self):
        # OpenCV BGR red pixel should become RGB [1, 0, 0] after preprocessing.
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        tensor = load_frame(frame)
        self.assertEqual(tuple(tensor.shape), (1, 3, 320, 320))
        self.assertTrue(torch.allclose(tensor[0, :, 100, 100], torch.tensor([1., 0., 0.]), atol=.01))

    def test_architecture_argument_is_required(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_legacy_architecture_selection_uses_legacy_adapter(self):
        # This checks architecture routing without opening a camera. A temporary
        # known-format checkpoint exercises the explicit legacy loader.
        import tempfile
        from pathlib import Path

        model = LegacyPersonDetector20()
        with tempfile.TemporaryDirectory(prefix="person-webcam-legacy-") as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"format_version": 1, "model_name": "PersonDetectorModel",
                        "state_dict": model.state_dict(), "class_names": ["person"],
                        "input_size": [320, 320]}, path)
            inference = build_inference("legacy20", str(path), "cpu", .1)
        self.assertTrue(inference.loaded)

    def test_annotation_draws_without_mutating_source(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        original = frame.copy()
        result = annotate_frame(frame, [{"bbox": [5, 6, 40, 50], "confidence": .8}], 12.3)
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, original))


if __name__ == "__main__":
    unittest.main()
