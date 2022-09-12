from torch import nn
import torch


class BaseNormalization(object):
    def __init__(self, mean, std):
        self.mean = nn.Parameter(torch.tensor(mean).reshape([1, -1, 1, 1]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).reshape([1, -1, 1, 1]), requires_grad=False)


class DeNormalization(BaseNormalization):
    def __call__(self, x):
        return self.std * x + self.mean


class Normalization(BaseNormalization):
    def __call__(self, x):
        return (x - self.mean) / self.std
