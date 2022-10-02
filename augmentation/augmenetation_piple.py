from torch import nn
from augmentation.normalization import DeNormalization, Normalization
from augmentation.dequantization import DeQuantization
from augmentation.mix_quant import MixedQuantized
from augmentation.lsb_noise import LSBNoise
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class AugmentationPipeline(object):
    def __init__(self, alpha=None, p=None, dequantization=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        self.norm = Normalization(mean, std)
        self.denorm = DeNormalization(mean, std)

        self.mix_quant = nn.Identity() if alpha is None else MixedQuantized(alpha)
        self.lsb_noise = nn.Identity() if p is None else LSBNoise(p)
        self.de_quant = nn.Identity() if not dequantization else DeQuantization()

    def __call__(self, x):
        x = self.denorm(x)
        x = self.mix_quant(x)
        x = self.lsb_noise(x)
        x = self.de_quant(x)
        return self.norm(x)


def generate_augmentation_function(mean, std, alpha=None, p=None, dequantization=True):
    return AugmentationPipeline(mean=mean, std=std, alpha=alpha, p=p, dequantization=dequantization)
