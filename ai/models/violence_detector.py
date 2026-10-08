"""Temporal binary classifier over a fixed sequence of RGB frames."""
import torch
import torch.nn as nn
from .common import FrameBackbone


class ViolenceClassifier(nn.Module):
    class_names = ["normal", "violence"]

    def __init__(self, width: int = 24):
        super().__init__()
        self.backbone = FrameBackbone(width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.GRU(self.backbone.out_channels, 64, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True),
                                        nn.Linear(64, len(self.class_names)))

    def forward(self, clips):
        # clips: [batch, time, channels, height, width]; GRU preserves frame order.
        if clips.ndim != 5:
            raise ValueError("clips must have shape [batch, time, channels, height, width]")
        batch, steps, channels, height, width = clips.shape
        if steps < 2:
            raise ValueError("violence classification requires at least two frames")
        if channels != 3 or min(height,width) < 16:
            raise ValueError("clip frames must be RGB with height and width at least 16")
        features = self.pool(self.backbone(clips.reshape(batch * steps, channels, height, width))).flatten(1)
        sequence = features.reshape(batch, steps, -1)
        _, hidden = self.temporal(sequence)
        return self.classifier(hidden[-1])
