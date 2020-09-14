from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .mobilenext import MobileNeXt
from .efficient_net import EfficientNet
from .i2rnet import I2RNet
from .se_mbv2 import SEMBV2
from .ganet import GANet

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'I2RNet', 'SEMBV2', 'GANet', 'MobileNeXt']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
