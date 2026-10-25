"""
Model registry. build_model(arch, num_classes, in_channels, image_size)
returns a ready-to-train nn.Module for any (dataset, architecture) pair
in the project's grid.

Two families of adaptation happen here, and only here:

1. ResNet family (10/18/50): the standard ImageNet stem (7x7 stride-2
   conv + maxpool) is replaced with a 3x3 stride-1 conv and no maxpool
   whenever image_size <= 64, exactly as done for the original CIFAR-10
   ResNet-18 in Phase 2.
2. VGG-19 / DenseNet-161: these assume much larger inputs and are
   harder to re-stem cleanly (VGG's 5 maxpools alone reduce a 32x32
   image to 1x1). Small inputs are upsampled to a minimum working
   resolution (64x64) internally instead — a pragmatic choice, not a
   claim that this is the optimal way to run these on tiny images.

There is no official "ResNet-10" — it's defined here the natural way
(BasicBlock, one block per stage, half of ResNet-18's depth).
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

SMALL_IMAGE_THRESHOLD = 64


def _adapt_resnet_stem(model, in_channels, image_size):
    if image_size <= SMALL_IMAGE_THRESHOLD:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                 stride=old.stride, padding=old.padding, bias=False)
    return model


class UpsampleWrapper(nn.Module):
    """Resizes small inputs up to `min_size` before the backbone. No-op
    if the input is already large enough."""

    def __init__(self, backbone, min_size):
        super().__init__()
        self.backbone = backbone
        self.min_size = min_size

    def forward(self, x):
        if x.shape[-1] < self.min_size or x.shape[-2] < self.min_size:
            x = F.interpolate(x, size=(self.min_size, self.min_size), mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_model(arch, num_classes, in_channels=3, image_size=32):
    arch = arch.lower()

    if arch == "resnet10":
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
        return _adapt_resnet_stem(model, in_channels, image_size)

    if arch == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        return _adapt_resnet_stem(model, in_channels, image_size)

    if arch == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        return _adapt_resnet_stem(model, in_channels, image_size)

    if arch == "vgg19":
        backbone = tvm.vgg19_bn(weights=None, num_classes=num_classes)
        if in_channels != 3:
            old = backbone.features[0]
            backbone.features[0] = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                              stride=old.stride, padding=old.padding)
        if image_size <= SMALL_IMAGE_THRESHOLD:
            return UpsampleWrapper(backbone, min_size=SMALL_IMAGE_THRESHOLD)
        return backbone

    if arch == "densenet161":
        backbone = tvm.densenet161(weights=None, num_classes=num_classes)
        if in_channels != 3:
            old = backbone.features.conv0
            backbone.features.conv0 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                                 stride=old.stride, padding=old.padding, bias=False)
        if image_size <= SMALL_IMAGE_THRESHOLD:
            return UpsampleWrapper(backbone, min_size=SMALL_IMAGE_THRESHOLD)
        return backbone

    raise ValueError(f"unknown architecture: {arch}")


MODEL_NAMES = ["resnet10", "resnet18", "resnet50", "vgg19", "densenet161"]
DATASET_NAMES = ["cifar10", "mnist", "gtsrb", "tinyimagenet", "imagenet_subset"]
