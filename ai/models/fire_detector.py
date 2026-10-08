"""Custom binary fire/normal image classifier."""
import torch.nn as nn
from .common import FrameBackbone


class FireClassifier(nn.Module):
    class_names = ["normal", "fire"]

    def __init__(self, width: int = 24):
        super().__init__()
        self.backbone = FrameBackbone(width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.out_channels, len(self.class_names))

    def forward(self, images):
        return self.classifier(self.pool(self.backbone(images)).flatten(1))
