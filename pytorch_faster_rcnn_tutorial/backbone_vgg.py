import torch
import logging
from enum import Enum
from typing import List, Optional, Tuple

import torchvision.models as models
from torchvision.models import vgg
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

logger: logging.Logger = logging.getLogger(__name__)

class VGGBackbones(Enum):
    VGG16 = "vgg16"

def vgg16_backbone(backbone_name : VGGBackbones, pretrained : bool) -> torch.nn.Sequential:
    print(backbone_name)
    print(VGGBackbones.VGG16)
    if backbone_name == VGGBackbones.VGG16:
        backbone = torchvision.models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        backbone = nn.Sequential(*list(backbone)[:-1])  # Remove the last max pooling layer
        backbone.out_channels = 512
    else:
        raise ValueError("Invalid backbone name.") 

    return backbone

def vgg16_anchor_generator() -> AnchorGenerator:
    anchor_sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,aspect_ratios=aspect_ratios)

    return anchor_generator

def vgg16_get_roi_pool(
    featmap_names: Optional[List[str]] = None,
    output_size: int = 7,
    sampling_ratio: int = 2,
) -> MultiScaleRoIAlign:
    """Returns the ROI Pooling"""
    if featmap_names is None:
        # default for vgg16
        featmap_names = ['0']


    roi_pooler = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=output_size,
        sampling_ratio=sampling_ratio,
    )

    return roi_pooler