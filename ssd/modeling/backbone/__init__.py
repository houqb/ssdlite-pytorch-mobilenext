from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .mobilenext import MobileNeXt


__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'MobileNeXt']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
