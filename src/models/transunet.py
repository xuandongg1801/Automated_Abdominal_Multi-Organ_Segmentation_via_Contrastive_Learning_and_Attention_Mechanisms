from __future__ import annotations

import torch
from torch import nn

from .layers import ChannelBridge, DecoderBlock, ResNet50Encoder, SegmentationHead


class PatchEmbedding(nn.Module):
    """Convert CNN feature maps into ViT-compatible token sequences."""

    def __init__(self, in_channels: int = 1024, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        return tokens


class TransformerToFeatureMap(nn.Module):
    """Reshape Transformer tokens back to a 2D feature map for the decoder."""

    def __init__(self, embed_dim: int = 768, out_channels: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = tokens.shape
        if num_tokens != height * width:
            raise ValueError(
                f"Cannot reshape {num_tokens} tokens to spatial size {height}x{width}."
            )
        x = tokens.transpose(1, 2).contiguous().view(batch_size, embed_dim, height, width)
        return self.proj(x)


class TimmVitEncoder(nn.Module):
    """Use timm ViT-B/16 Transformer blocks without the image patch stem."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_first_n_blocks: int = 0,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "TransUNet requires timm. On Kaggle, install it with: pip install timm"
            ) from exc

        vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = getattr(vit, "embed_dim", 768)
        self.blocks = vit.blocks
        self.norm = vit.norm

        for block in self.blocks[:freeze_first_n_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class TransUNet(nn.Module):
    """Hybrid ResNet-50 + ViT-B/16 + U-Net decoder segmentation model."""

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 3,
        pretrained_encoder: bool = True,
        pretrained_transformer: bool = True,
        transformer_model_name: str = "vit_base_patch16_224",
        freeze_first_n_transformer_blocks: int = 6,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder(
            in_channels=in_channels,
            pretrained=pretrained_encoder,
            keep_layer4=False,
        )
        self.patch_embedding = PatchEmbedding(in_channels=1024, embed_dim=768)
        self.position_embedding = nn.Parameter(torch.zeros(1, 14 * 14, 768))
        self.transformer = TimmVitEncoder(
            model_name=transformer_model_name,
            pretrained=pretrained_transformer,
            freeze_first_n_blocks=freeze_first_n_transformer_blocks,
        )
        self.token_to_map = TransformerToFeatureMap(embed_dim=768, out_channels=512)

        self.bridge2 = ChannelBridge(512, 256)
        self.bridge1 = ChannelBridge(256, 128)
        self.bridge0 = ChannelBridge(64, 64)

        self.decoder1 = DecoderBlock(512, 256, skip_channels=256, dropout=decoder_dropout)
        self.decoder2 = DecoderBlock(256, 128, skip_channels=128, dropout=decoder_dropout)
        self.decoder3 = DecoderBlock(128, 64, skip_channels=64, dropout=decoder_dropout)
        self.decoder4 = DecoderBlock(64, 64, skip_channels=0, dropout=decoder_dropout)
        self.head = SegmentationHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, include_layer4=False)
        bottleneck = features["layer3"]
        height, width = bottleneck.shape[-2:]

        tokens = self.patch_embedding(bottleneck)
        if tokens.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                "TransUNet expects 224x224 inputs so the CNN bottleneck is 14x14. "
                f"Got bottleneck size {height}x{width}."
            )
        tokens = tokens + self.position_embedding
        tokens = self.transformer(tokens)
        x = self.token_to_map(tokens, height=height, width=width)

        x = self.decoder1(x, self.bridge2(features["layer2"]))
        x = self.decoder2(x, self.bridge1(features["layer1"]))
        x = self.decoder3(x, self.bridge0(features["stem"]))
        x = self.decoder4(x)
        return self.head(x)
