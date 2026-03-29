import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TesseraSegmentationModel(nn.Module):
    """
    Lightweight TESSERA-style wrapper with:
    - channel adapter for Sentinel inputs
    - encoder trunk
    - trainable segmentation head

    This provides a stable interface now and can later be swapped to official
    TESSERA encoder weights with minimal pipeline changes.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        adapter_channels: int = 3,
        head_hidden_channels: int = 512,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "adapter_channels": adapter_channels,
            "head_hidden_channels": head_hidden_channels,
            "freeze_encoder": freeze_encoder,
        }

        self.channel_adapter = nn.Conv2d(in_channels, adapter_channels, kernel_size=1)

        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.seg_head = SegmentationHead(
            in_channels=2048,
            hidden_channels=head_hidden_channels,
            num_classes=num_classes,
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks: int = 1) -> None:
        blocks = [self.layer4, self.layer3, self.layer2, self.layer1]
        for block in blocks[: max(0, num_blocks)]:
            for param in block.parameters():
                param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapter(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.forward_features(x)
        logits = self.seg_head(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_type": "tessera",
            "config": self.config,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(cls, path: str, map_location: Optional[str] = None) -> "TesseraSegmentationModel":
        payload = torch.load(path, map_location=map_location)

        if isinstance(payload, dict) and "state_dict" in payload and "config" in payload:
            model = cls(**payload["config"])
            model.load_state_dict(payload["state_dict"])
            return model

        # Fallback for raw state_dict checkpoints
        model = cls()
        model.load_state_dict(payload)
        return model
