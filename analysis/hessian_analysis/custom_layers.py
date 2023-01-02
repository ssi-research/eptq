import torch
from torch import nn


class Add(nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)
