import torch
from torch import nn
from augmentation.normalization import DeNormalization, Normalization
from augmentation.dequantization import DeQuantization
from augmentation.mix_quant import MixedQuantized
from augmentation.lsb_noise import LSBNoise
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class AugmentationPipeline(object):
    def __init__(self, alpha=0.5, p=0.5, dequantization=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        self.norm = Normalization(mean, std)
        self.denorm = DeNormalization(mean, std)
        self.mix_quant = MixedQuantized(alpha)
        self.lsb_noise = LSBNoise(p)
        if dequantization:
            self.de_quant = DeQuantization()
        else:
            self.de_quant = nn.Identity()

    def __call__(self, x):
        x_int = self.denorm(x)
        x_mix = self.mix_quant(x_int)
        x_lsb = self.lsb_noise(x_mix)
        x_dequant = self.de_quant(x_lsb)
        return self.norm(x_dequant)


def generate_augmentation_function(mean, std, alpha=0.5, p=0.5, dequantization=True):
    return AugmentationPipeline(mean, std, alpha, p, dequantization)
