"""Segmentation head on precomputed TESSERA embeddings; tessera_embed_train trains it, tessera_embed_infer loads it."""
from typing import Dict

import torch
import torch.nn as nn


class TesseraEmbeddingSegHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_classes: int = 4, hidden_channels: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"out": self.head(x)}
