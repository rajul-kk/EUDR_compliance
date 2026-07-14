"""M3: Siamese-DeepLabV3 change detection model (Late Fusion).

Shared ResNet50 encoder processes t1 and t2 independently.
Change features are computed at three scales (layer2/3/4) as
cat([f1, f2, |f1-f2|]) and fused top-down via a lightweight FPN before
feeding into an ASPP change head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50  # noqa: F401


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


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


class _ChangeASPP(nn.Module):
    """ASPP operating on the fused change feature tensor."""

    def __init__(self, in_channels: int, num_classes: int, dropout_p: float) -> None:
        super().__init__()
        out_ch = 256
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            _ASPPConv(in_channels, out_ch, 6),
            _ASPPConv(in_channels, out_ch, 12),
            _ASPPConv(in_channels, out_ch, 18),
            _ASPPPooling(in_channels, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.classifier = nn.Conv2d(out_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.project(torch.cat([c(x) for c in self.convs], dim=1)))


class _FPNDiff(nn.Module):
    """Multi-scale change features fused via a top-down FPN.

    At each scale the change feature is cat([f1, f2, |f1-f2|]):
      layer2 → 3×512  = 1536 ch  at H/8
      layer3 → 3×1024 = 3072 ch  at H/16
      layer4 → 3×2048 = 6144 ch  at H/32

    Each scale is projected to 256 ch, then fused top-down (l4→l3→l2).
    A smoothing conv stabilises the summed feature map.
    Output: 256 ch at H/8 (layer2 resolution).
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj4 = _proj(3 * 2048, 256)
        self.proj3 = _proj(3 * 1024, 256)
        self.proj2 = _proj(3 * 512,  256)
        self.smooth = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        p4 = self.proj4(c4)
        p3 = self.proj3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.proj2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(p2)  # (B, 256, H/8, W/8)


class SiameseDeepLabV3(nn.Module):
    """Siamese DeepLabV3 for binary forest-loss change detection.

    Shared ResNet50 encoder extracts features at three scales (layer2/3/4).
    Per-scale change features cat([f1, f2, |f1-f2|]) are fused top-down via
    a lightweight FPN, then decoded by an ASPP change head.
    """

    def __init__(self, in_channels: int = 7, num_classes: int = 2, dropout_p: float = 0.2) -> None:
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            new_conv1.weight[:, :3] = backbone.conv1.weight
        backbone.conv1 = new_conv1

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn         = _FPNDiff()
        self.change_head = _ChangeASPP(in_channels=256, num_classes=num_classes, dropout_p=dropout_p)

    def _encode(self, x: torch.Tensor):
        x  = self.stem(x)
        x  = self.layer1(x)
        l2 = self.layer2(x)   # (B, 512,  H/8,  W/8)
        l3 = self.layer3(l2)  # (B, 1024, H/16, W/16)
        l4 = self.layer4(l3)  # (B, 2048, H/32, W/32)
        return l2, l3, l4

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        input_size = t1.shape[-2:]

        l2_1, l3_1, l4_1 = self._encode(t1)
        l2_2, l3_2, l4_2 = self._encode(t2)

        # Per-scale change features: cat([f1, f2, |f1-f2|])
        c2 = torch.cat([l2_1, l2_2, torch.abs(l2_1 - l2_2)], dim=1)  # 1536 ch
        c3 = torch.cat([l3_1, l3_2, torch.abs(l3_1 - l3_2)], dim=1)  # 3072 ch
        c4 = torch.cat([l4_1, l4_2, torch.abs(l4_1 - l4_2)], dim=1)  # 6144 ch

        fused  = self.fpn(c2, c3, c4)          # (B, 256, H/8, W/8)
        logits = self.change_head(fused)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}


def get_siamese_model(in_channels: int = 7, num_classes: int = 2, dropout_p: float = 0.2) -> SiameseDeepLabV3:
    return SiameseDeepLabV3(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
