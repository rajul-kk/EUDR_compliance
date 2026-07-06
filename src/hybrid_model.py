"""M6: DeepLabV3 + tessera-embed context hybrid.

Pixel-level Sentinel-2 features (from ResNet50 encoder) are concatenated with
the GeoTESSERA 128-dim regional embedding — broadcast spatially — before the
ASPP decoder. The embed acts as a "where on Earth" prior: a cocoa farm in Ivory
Coast and a meadow in Iowa can share similar NDVI yet have completely different
landscape-scale embedding vectors.

This is a two-stage model (still runs on t1 and t2 separately). It improves
M1a accuracy without changing the pipeline structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)


class DeepLabWithEmbedContext(nn.Module):
    """DeepLabV3 encoder + broadcast tessera-embed context + ASPP decoder.

    Args:
        in_channels:   Number of Sentinel-2 input channels (default 7 with NDWI).
        embed_channels: Dimension of tessera-embed vector (default 128).
        num_classes:   Number of segmentation classes (default 3 for Hansen: non-forest/forest-2020/post-loss).
        dropout_p:     Dropout probability in ASPP head (used for MC Dropout).
    """

    def __init__(
        self,
        in_channels: int = 7,
        embed_channels: int = 128,
        num_classes: int = 3,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            new_conv1.weight[:, :3] = backbone.conv1.weight
        backbone.conv1 = new_conv1

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Project embed vector to match encoder feature dim before concat
        self.embed_proj = nn.Sequential(
            nn.Conv2d(embed_channels, embed_channels, 1, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        fused_channels = 2048 + embed_channels
        out_ch = 256
        self.aspp = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fused_channels, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            _ASPPConv(fused_channels, out_ch, 6),
            _ASPPConv(fused_channels, out_ch, 12),
            _ASPPConv(fused_channels, out_ch, 18),
            _ASPPPooling(fused_channels, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.classifier = nn.Conv2d(out_ch, num_classes, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)  # (B, 2048, H', W')

    def forward(self, image: torch.Tensor, embed: torch.Tensor):
        """
        Args:
            image: (B, in_channels, H, W) Sentinel-2 composite.
            embed: (B, embed_channels) GeoTESSERA vector per sample — broadcast spatially.
        """
        input_size = image.shape[-2:]
        features = self._encode(image)          # (B, 2048, H', W')
        h, w = features.shape[-2:]

        # Broadcast embed: (B, 128) → (B, 128, 1, 1) → (B, 128, H', W')
        embed_spatial = embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        embed_spatial = self.embed_proj(embed_spatial)

        fused = torch.cat([features, embed_spatial], dim=1)  # (B, 2176, H', W')

        logits = self.classifier(self.project(torch.cat([c(fused) for c in self.aspp], dim=1)))
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}


def get_hybrid_model(in_channels: int = 7, embed_channels: int = 128,
                     num_classes: int = 3, dropout_p: float = 0.2) -> DeepLabWithEmbedContext:
    return DeepLabWithEmbedContext(in_channels=in_channels, embed_channels=embed_channels,
                                   num_classes=num_classes, dropout_p=dropout_p)
