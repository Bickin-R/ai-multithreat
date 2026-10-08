"""Tests for Person positive/negative objectness balancing."""
import unittest

import torch
import torch.nn.functional as F

from ai.training.train_utils import (_balanced_person_objectness_loss,
                                     _detection_loss, _make_targets)


class ObjectnessLossTests(unittest.TestCase):
    def test_positive_weight_uses_square_root_of_negative_positive_ratio(self):
        logits = torch.zeros(1, 1, 1, 100, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[..., 0] = 1.0
        loss = _balanced_person_objectness_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        positive_gradient = logits.grad[..., 0].abs().sum()
        negative_gradient = logits.grad[..., 1:].abs().sum()
        expected_ratio = 1.0 / (99.0 ** 0.5)
        self.assertAlmostEqual(float(positive_gradient / negative_gradient), expected_ratio, places=6)

        raw_losses = F.binary_cross_entropy_with_logits(logits.detach(), targets, reduction="none")
        positive_weight = 99.0 ** 0.5
        expected_loss = (raw_losses[..., 1:].sum() + positive_weight * raw_losses[..., 0].sum()) / (99.0 + positive_weight)
        self.assertTrue(torch.allclose(loss.detach(), expected_loss))

    def test_empty_positive_batch_falls_back_to_mean_bce(self):
        logits = torch.randn(2, 2, 20, 20, requires_grad=True)
        targets = torch.zeros_like(logits)
        loss = _balanced_person_objectness_loss(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_weapon_detection_loss_keeps_original_unweighted_bce(self):
        pred = torch.randn(1, 10, 4, 4, requires_grad=True)
        objects = [torch.tensor([[2, .3, .4, .2, .25]], dtype=torch.float32)]
        loss = _detection_loss(pred, objects, class_count=5)
        target, positive, target_classes = _make_targets(pred, objects, class_count=5)
        object_loss = F.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0, 0])
        box_pred = pred[:, 1:5].sigmoid().permute(0, 2, 3, 1)[positive[:, 0]]
        box_true = target[:, 0, 1:5].permute(0, 2, 3, 1)[positive[:, 0]]
        box_loss = F.smooth_l1_loss(box_pred, box_true)
        class_logits = pred[:, 5:].permute(0, 2, 3, 1)[positive[:, 0]]
        class_loss = F.cross_entropy(class_logits, target_classes[:, 0][positive[:, 0]])
        self.assertTrue(torch.allclose(loss, object_loss + box_loss + class_loss))


if __name__ == "__main__":
    unittest.main()
