"""Synthetic mechanics tests for the custom two-slot Person detector."""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from ai.inference.common import decode_grid, load_checkpoint
from ai.models import LegacyPersonDetector20, PersonDetectorModel
from ai.models.common import FrameBackbone
from ai.training.train_utils import (_detection_loss, _make_targets,
                                     checkpoint_payload, load_model_checkpoint)


class PersonDetectorSlotTests(unittest.TestCase):
    def test_model_outputs_two_five_value_slots_per_grid_cell(self):
        model = PersonDetectorModel().eval()
        with torch.no_grad():
            output = model(torch.rand(2, 3, 320, 320))
        self.assertEqual(tuple(output.shape), (2, 10, 40, 40))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 329266)

    def test_two_objects_can_share_a_cell_and_third_is_rejected(self):
        pred = torch.zeros(1, 10, 40, 40)
        two = [torch.tensor([[0, .501, .502, .1, .2],
                             [0, .510, .512, .1, .2]], dtype=torch.float32)]
        target, positive, _ = _make_targets(pred, two, class_count=1)
        self.assertEqual(tuple(positive.shape), (1, 2, 40, 40))
        self.assertEqual(int(positive[0, :, 20, 20].sum()), 2)
        self.assertEqual(float(target[0, 0, 0, 20, 20]), 1.0)
        self.assertEqual(float(target[0, 1, 0, 20, 20]), 1.0)
        three = [torch.cat((two[0], torch.tensor([[0, .515, .518, .1, .2]])))]
        with self.assertRaisesRegex(ValueError, "supports 2 per cell"):
            _make_targets(pred, three, class_count=1)

    def test_collision_loss_is_finite_and_backpropagates(self):
        pred = torch.randn(1, 10, 40, 40, requires_grad=True)
        objects = [torch.tensor([[0, .501, .502, .1, .2],
                                 [0, .510, .512, .1, .2]], dtype=torch.float32)]
        loss = _detection_loss(pred, objects, class_count=1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(pred.grad.abs().sum()), 0.0)

    def test_decoder_emits_both_person_slots_from_one_cell(self):
        raw = torch.full((1, 10, 40, 40), -12.0)
        raw[0, 0, 20, 20] = 12
        raw[0, 5, 20, 20] = 12
        raw[0, 1:5, 20, 20] = torch.logit(torch.tensor([.505, .525, .02, .2]))
        raw[0, 6:10, 20, 20] = torch.logit(torch.tensor([.545, .525, .02, .2]))
        detections = decode_grid(raw, (320, 320, 3), ["person"], threshold=.5)
        self.assertEqual(len(detections), 2)
        self.assertTrue(all(item["class"] == "person" for item in detections))

    def test_checkpoint_metadata_identifies_person_grid(self):
        payload = checkpoint_payload(PersonDetectorModel())
        self.assertEqual(payload["architecture_id"], "person_stride8_grid40_two_slot_v1")
        self.assertEqual(payload["output_grid"], [40, 40])
        with tempfile.TemporaryDirectory(prefix="person-grid40-checkpoint-") as tmp:
            checkpoint = Path(tmp) / "person_grid40.pt"
            torch.save(payload, checkpoint)
            load_model_checkpoint(PersonDetectorModel(), checkpoint)
            self.assertTrue(load_checkpoint(PersonDetectorModel(), checkpoint, "cpu"))

    def test_old_twenty_grid_person_checkpoint_is_rejected(self):
        old_model = nn.Module()
        old_model.backbone = FrameBackbone(24, block7_stride=2)
        old_model.head = nn.Conv2d(old_model.backbone.out_channels, 10, 1)
        payload = {"format_version": 1, "model_name": "PersonDetectorModel",
                   "state_dict": old_model.state_dict(), "class_names": ["person"],
                   "input_size": [320, 320]}
        with tempfile.TemporaryDirectory(prefix="person-old-checkpoint-") as tmp:
            checkpoint = Path(tmp) / "old.pt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "architecture_id"):
                load_model_checkpoint(PersonDetectorModel(), checkpoint)
            with self.assertRaisesRegex(ValueError, "architecture_id"):
                load_checkpoint(PersonDetectorModel(), checkpoint, "cpu")

    def test_legacy20_and_current40_checkpoint_paths_are_explicit(self):
        from ai.inference.common import load_checkpoint
        from ai.inference.legacy_person_inference import load_legacy_person_checkpoint

        legacy = LegacyPersonDetector20().eval()
        current = PersonDetectorModel().eval()
        with tempfile.TemporaryDirectory(prefix="person-architecture-checkpoints-") as tmp:
            legacy_path = Path(tmp) / "legacy20.pt"
            current_path = Path(tmp) / "current40.pt"
            torch.save({"format_version": 1, "model_name": "PersonDetectorModel",
                        "state_dict": legacy.state_dict(), "class_names": ["person"],
                        "input_size": [320, 320]}, legacy_path)
            torch.save(checkpoint_payload(current), current_path)

            loaded_legacy = load_legacy_person_checkpoint(LegacyPersonDetector20(), legacy_path)
            with self.assertRaisesRegex(ValueError, "architecture_id"):
                load_checkpoint(PersonDetectorModel(), legacy_path, "cpu")
            self.assertTrue(load_checkpoint(PersonDetectorModel(), current_path, "cpu"))
            self.assertEqual(tuple(loaded_legacy(torch.rand(2, 3, 320, 320)).shape),
                             (2, 10, 20, 20))
            self.assertEqual(tuple(current(torch.rand(2, 3, 320, 320)).shape),
                             (2, 10, 40, 40))


if __name__ == "__main__":
    unittest.main()
