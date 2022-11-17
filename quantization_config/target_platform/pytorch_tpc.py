import operator
import torch
from torch import add, subtract, multiply, divide, flatten, reshape, split, unsqueeze, dropout, sigmoid, tanh
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn import Dropout, Flatten, Hardtanh
from torch.nn import ReLU, ReLU6, PReLU, SiLU, Sigmoid, Tanh, Hardswish, Hardsigmoid
from torch.nn.functional import relu, relu6, prelu, silu, hardtanh, hardswish, hardsigmoid
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.reader.graph_builders import DummyPlaceHolder

tp = mct.target_platform


def generate_pytorch_tpc(tp_model: tp.TargetPlatformModel, name: str):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.

    Args:
        name: Name of the TargetPlatformCapabilities.
        tp_model: TargetPlatformModel object.

    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    torch_tpc = tp.TargetPlatformCapabilities(tp_model, name=name)
    with torch_tpc:
        tp.OperationsSetToLayers("NoQuantization",   [Dropout,
                                                      Flatten,
                                                      dropout,
                                                      flatten,
                                                      split,
                                                      operator.getitem,
                                                      reshape,
                                                      unsqueeze,
                                                      BatchNorm2d,
                                                      torch.Tensor.size])

        tp.OperationsSetToLayers("Conv", [Conv2d])
        tp.OperationsSetToLayers("FullyConnected", [Linear])
        tp.OperationsSetToLayers("AnyReLU", [torch.relu,
                                             ReLU,
                                             ReLU6,
                                             relu,
                                             relu6,
                                             tp.LayerFilterParams(Hardtanh, min_val=0),
                                             tp.LayerFilterParams(hardtanh, min_val=0)])
        tp.OperationsSetToLayers("Add", [operator.add, add])
        tp.OperationsSetToLayers("Sub", [operator.sub, subtract])
        tp.OperationsSetToLayers("Mul", [operator.mul, multiply])
        tp.OperationsSetToLayers("Div", [operator.truediv, divide])
        tp.OperationsSetToLayers("PReLU", [PReLU, prelu])
        tp.OperationsSetToLayers("Swish", [SiLU, silu, Hardswish, hardswish])
        tp.OperationsSetToLayers("Sigmoid", [Sigmoid, sigmoid])
        tp.OperationsSetToLayers("Tanh", [Tanh, tanh])
        tp.OperationsSetToLayers("Input", [DummyPlaceHolder])

    return torch_tpc
