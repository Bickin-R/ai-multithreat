"""Custom grid-based person detector with two object/box slots per cell."""
import torch.nn as nn
from .common import FrameBackbone


class PersonDetectorModel(nn.Module):
    class_names = ["person"]
    architecture_id = "person_stride8_grid40_two_slot_v1"
    output_grid = (40, 40)

    def __init__(self, width: int = 24):
        super().__init__()
        self.backbone = FrameBackbone(width, block7_stride=1)
        # Packed slot channels: [slot0: objectness,cx,cy,w,h,
        #                        slot1: objectness,cx,cy,w,h].
        self.head = nn.Conv2d(self.backbone.out_channels, 2 * 5, 1)

    def forward(self, images):
        return self.head(self.backbone(images))
