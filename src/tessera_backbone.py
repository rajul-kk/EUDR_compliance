import os
from typing import Any, Dict, Optional

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
        size = x.shape[-2:]
        return F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)


class ASPPHead(nn.Module):
    """ASPP segmentation head matching DeepLabV3's multi-scale pooling."""

    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.2) -> None:
        super().__init__()
        out_ch = 256
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
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


class TesseraSegmentationModel(nn.Module):
    """
    TESSERA-style wrapper: ImageNet-pretrained ResNet50 encoder + ASPP head.

    The 6-channel first conv replaces the original 3-channel one so all
    Sentinel-2 bands (R, G, B, NIR, SCL, NDVI) propagate at full encoder depth.
    Swap `weights=ResNet50_Weights.DEFAULT` for official TESSERA encoder weights
    once they are publicly released — the rest of the pipeline is unchanged.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        freeze_encoder: bool = True,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "freeze_encoder": freeze_encoder,
            "dropout_p": dropout_p,
        }

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace 3-channel first conv with in_channels-channel version.
        # Kaiming init preserves gradient flow; RGB channels copy pretrained weights.
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            new_conv1.weight[:, :3] = backbone.conv1.weight  # copy pretrained RGB weights
        backbone.conv1 = new_conv1

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.seg_head = ASPPHead(in_channels=2048, num_classes=num_classes, dropout_p=dropout_p)

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks: int = 1) -> None:
        for block in [self.layer4, self.layer3, self.layer2, self.layer1][:max(0, num_blocks)]:
            for param in block.parameters():
                param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        logits = self.seg_head(self.forward_features(x))
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"model_type": "tessera", "config": self.config, "state_dict": self.state_dict(), "extra": extra or {}},
            path,
        )

    @classmethod
    def load_from_checkpoint(cls, path: str, map_location: Optional[str] = None) -> "TesseraSegmentationModel":
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload and "config" in payload:
            model = cls(**payload["config"])
            model.load_state_dict(payload["state_dict"])
            return model
        model = cls()
        model.load_state_dict(payload)
        return model
