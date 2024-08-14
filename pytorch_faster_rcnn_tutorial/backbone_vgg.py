import torch
import torchvision.models as models
from torchvision.models.detection import FasterRCNN

logger: logging.Logger = logging.getLogger(__name__)

def vgg16_backbone() -> torch.nn.Sequential:
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT, progress=False)
    backbone = vgg16.features
    backbone.out_channels = 512

    return backbone

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
aspect_ratios=((0.5, 1.0, 2.0),))