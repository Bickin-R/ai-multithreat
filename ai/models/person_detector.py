"""Custom grid-based person detector: objectness and normalized box per cell."""
import torch.nn as nn
from .common import FrameBackbone


class PersonDetectorModel(nn.Module):
    class_names = ["person"]

    def __init__(self, width: int = 24):
        super().__init__()
        self.backbone = FrameBackbone(width)
        self.head = nn.Conv2d(self.backbone.out_channels, 5, 1)  # objectness, cx, cy, w, h

    def forward(self, images):
        return self.head(self.backbone(images))
