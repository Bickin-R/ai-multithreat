"""Small CNN building blocks; all layers initialize from scratch."""
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class FrameBackbone(nn.Module):
    """Compact feature extractor, initialized without pretrained weights."""
    def __init__(self, width: int = 24, block7_stride: int = 2):
        super().__init__()
        self.layers = nn.Sequential(ConvBlock(3, width, 2), ConvBlock(width, width, 1),
                                    ConvBlock(width, width * 2, 2), ConvBlock(width * 2, width * 2, 1),
                                    ConvBlock(width * 2, width * 4, 2), ConvBlock(width * 4, width * 4, 1),
                                    ConvBlock(width * 4, width * 4, block7_stride),
                                    ConvBlock(width * 4, width * 4, 1))
        self.out_channels = width * 4

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("model input must have shape [batch, 3, height, width]")
        if min(x.shape[-2:]) < 16:
            raise ValueError("image height and width must each be at least 16 pixels")
        return self.layers(x)
