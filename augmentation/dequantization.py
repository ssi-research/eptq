import torch

DELTA = 1 / 2 ** 8


class DeQuantization(object):
    def __init__(self, delta=DELTA):
        self.delta = delta

    def __call__(self, x):
        dequantization_noise = self.delta * torch.rand(x.shape) - self.delta / 2
        return x + dequantization_noise
