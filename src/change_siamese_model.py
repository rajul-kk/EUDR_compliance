"""M3: Siamese-DeepLabV3 change detection model (Late Fusion).

Shared ResNet50 encoder processes t1 and t2 independently.
Element-wise feature difference at the ASPP bottleneck captures change in
feature space — more robust to cross-year radiometric shift than early fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50


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
    """ASPP operating on the feature-difference tensor."""

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


class SiameseDeepLabV3(nn.Module):
    """Siamese DeepLabV3 for binary forest-loss change detection.

    A single shared ResNet50 encoder (pretrained, weights shared) processes
    t1 and t2 separately. The element-wise absolute difference of their deep
    features feeds into an ASPP change head that outputs a 2-class change map.

    The shared-weight constraint means the same semantic "eye" looks at both
    time periods — the feature difference is invariant to per-image scale and
    captures genuine semantic change rather than style differences.
    """

    def __init__(self, in_channels: int = 7, num_classes: int = 2, dropout_p: float = 0.2) -> None:
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace first conv to handle in_channels input bands
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            new_conv1.weight[:, :3] = backbone.conv1.weight
        backbone.conv1 = new_conv1

        # Shared encoder — single module, called twice per forward pass
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Change head operates on cat([f1, f2, |f1-f2|]) — retains direction of change
        self.change_head = _ChangeASPP(in_channels=2048 * 3, num_classes=num_classes, dropout_p=dropout_p)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 2048, H/8, W/8) with dilated conv not applied — simple stride

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        input_size = t1.shape[-2:]
        f1 = self._encode(t1)
        f2 = self._encode(t2)

        # Concatenate both feature maps and their absolute difference:
        # f1 and f2 supply directionality (gain vs. loss); |f1-f2| supplies magnitude.
        diff = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)  # (B, 6144, H', W')

        logits = self.change_head(diff)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}


def get_siamese_model(in_channels: int = 7, num_classes: int = 2, dropout_p: float = 0.2) -> SiameseDeepLabV3:
    return SiameseDeepLabV3(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
