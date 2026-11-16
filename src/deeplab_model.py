"""DeepLabV3-ResNet50 for N-band Sentinel-2 input; ML_farm_net trains it, inference loads it."""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50


def get_deeplab_model(num_classes: int, in_channels: int = 7, rgb_from_imagenet: bool = True) -> nn.Module:
    """A new first convolution for `in_channels` bands; its RGB filters start from ImageNet when `rgb_from_imagenet`
    (training), random otherwise (a model whose weights will be loaded)."""
    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
    if rgb_from_imagenet:
        with torch.no_grad():
            new_conv1.weight[:, :3] = resnet50(weights=ResNet50_Weights.DEFAULT).conv1.weight
    model.backbone.conv1 = new_conv1
    return model
