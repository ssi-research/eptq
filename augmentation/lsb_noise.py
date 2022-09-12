import torch

DELTA = 1 / 2 ** 8


class LSBNoise(object):
    def __init__(self, p_zero, delta=DELTA):
        self.shift = 0.5 - p_zero
        self.delta = delta

    def __call__(self, x):
        lsb_noise = torch.round(3 * torch.rand(x.shape) - 1.5)
        return x + self.delta * lsb_noise
