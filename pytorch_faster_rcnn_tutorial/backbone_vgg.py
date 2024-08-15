import torch
from typing import List, Optional, Tuple

import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

logger: logging.Logger = logging.getLogger(__name__)

def vgg16_backbone() -> torch.nn.Sequential:
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT, progress=False)
    backbone = vgg16.features
    backbone.out_channels = 512

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