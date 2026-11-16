"""DeepLabV3 atrous spatial pyramid pooling branches, shared by the TESSERA, hybrid and Siamese heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
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
