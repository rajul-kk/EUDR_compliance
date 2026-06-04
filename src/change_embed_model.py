"""M5: tessera-embed change detection head.

Accepts element-wise difference of two 128-dim embedding tensors and outputs
a binary change map (0=no change, 1=forest loss). ~200K params, CPU-only.
"""

import torch
import torch.nn as nn


class ChangeEmbedHead(nn.Module):
    """Lightweight segmentation head for embedding-space change detection."""

    def __init__(self, in_channels: int = 128, num_classes: int = 2, hidden_channels: int = 256, dropout_p: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(hidden_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        return {"out": self.head(x)}


def get_change_embed_model(in_channels: int = 128, num_classes: int = 2,
                           hidden_channels: int = 256, dropout_p: float = 0.2) -> ChangeEmbedHead:
    return ChangeEmbedHead(in_channels=in_channels, num_classes=num_classes,
                           hidden_channels=hidden_channels, dropout_p=dropout_p)
