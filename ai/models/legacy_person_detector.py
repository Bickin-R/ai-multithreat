"""Inference-only custom Person CNN for legacy 20x20 checkpoints."""
import torch.nn as nn

from .common import FrameBackbone


class LegacyPersonDetector20(nn.Module):
    """The original two-slot, stride-16 Person model used by legacy checkpoints.

    This class exists only to run explicitly selected 20x20 inference
    checkpoints. New training uses :class:`PersonDetectorModel`.
    """

    class_names = ["person"]
    architecture_id = "person_stride16_grid20_two_slot_legacy_v1"
    output_grid = (20, 20)

    def __init__(self, width: int = 24):
        super().__init__()
        self.backbone = FrameBackbone(width, block7_stride=2)
        self.head = nn.Conv2d(self.backbone.out_channels, 2 * 5, 1)

    def forward(self, images):
        return self.head(self.backbone(images))
