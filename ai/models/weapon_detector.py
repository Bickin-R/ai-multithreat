"""Custom grid-based detector for five weapon classes."""
import torch.nn as nn
from .common import FrameBackbone


class WeaponDetectorModel(nn.Module):
    class_names = ["knife", "gun", "arivaal", "baseball_bat", "other_weapon"]

    def __init__(self, width: int = 24):
        super().__init__()
        self.backbone = FrameBackbone(width)
        self.head = nn.Conv2d(self.backbone.out_channels, 5 + len(self.class_names), 1)

    def forward(self, images):
        return self.head(self.backbone(images))
