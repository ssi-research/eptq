from analysis.hessian_analysis.mobilenet_v2_modifed import mobilenet_v2, MobileNet_V2_Weights
from analysis.hessian_analysis.resnet_modifed import resnet18, resnet50, ResNet50_Weights, ResNet18_Weights
from enum import Enum


class Model(Enum):
    ResNet18 = 0
    ResNet50W1 = 1
    ResNet50W2 = 2
    MobileNetV2W1 = 3
    MobileNetV2W2 = 4


MODEL_DICT = {Model.ResNet18: resnet18,
              Model.ResNet50W1: resnet50,
              Model.ResNet50W2: resnet50,
              Model.MobileNetV2W1: mobilenet_v2,
              Model.MobileNetV2W2: mobilenet_v2}


def get_model(model_type: Model, pretrained, device):
    name = model_type.name
    if pretrained:
        name += "_pretrain"
    model_fn = MODEL_DICT.get(model_type)
    weights = None
    if model_type == Model.ResNet18:
        weights = ResNet18_Weights.IMAGENET1K_V1
    elif model_type == Model.ResNet50W1:
        weights = ResNet50_Weights.IMAGENET1K_V1
    elif model_type == Model.ResNet50W2:
        weights = ResNet50_Weights.IMAGENET1K_V2
    elif model_type == Model.MobileNetV2W1:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    elif model_type == Model.MobileNetV2W2:
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
    if not pretrained:
        weights = None
    net = model_fn(weights=weights).to(device)
    return net, name
