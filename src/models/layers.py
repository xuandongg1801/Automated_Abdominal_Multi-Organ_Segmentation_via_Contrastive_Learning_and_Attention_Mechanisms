from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from ..utils import adapt_first_conv, get_resnet50


class ConvBnRelu(nn.Sequential):
    """A compact Conv2d -> BatchNorm2d -> ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    """Two ConvBnRelu blocks, matching the U-Net decoder convention."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        layers: "OrderedDict[str, nn.Module]" = OrderedDict()
        layers["conv1"] = ConvBnRelu(in_channels, out_channels)
        layers["conv2"] = ConvBnRelu(out_channels, out_channels)
        if dropout > 0:
            layers["dropout"] = nn.Dropout2d(dropout)
        super().__init__(layers)


class ChannelBridge(nn.Sequential):
    """Project encoder skip channels before concatenating with decoder features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample, optionally concatenate a skip map, then refine with double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    "Decoder skip shape mismatch: "
                    f"upsampled={tuple(x.shape[-2:])}, skip={tuple(skip.shape[-2:])}"
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationHead(nn.Conv2d):
    """A 1x1 projection from decoder features to per-class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(in_channels, num_classes, kernel_size=1)


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature extractor with spatial skips for U-Net style decoders."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        keep_layer4: bool = True,
    ) -> None:
        super().__init__()
        resnet = get_resnet50(pretrained=pretrained)
        if in_channels != 3:
            adapt_first_conv(resnet, in_channels=in_channels)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 if keep_layer4 else None

    def forward(self, x: torch.Tensor, include_layer4: bool = True) -> dict[str, torch.Tensor]:
        stem = self.stem(x)
        pooled = self.maxpool(stem)
        layer1 = self.layer1(pooled)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        features = {
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
        }
        if include_layer4:
            if self.layer4 is None:
                raise ValueError("This encoder was created without layer4.")
            features["layer4"] = self.layer4(layer3)
        return features
