import timm
from models.pytorch_model_config import ModelParameters
from timm.models.resnet import resnet18, resnet50
from timm.models.regnet import regnetx_006, regnetx_032
from datasets.image_utils import pytorch_model_accuracy_evaluation


pytorch_model_dictionary = {
    'resnet18': ModelParameters(
        model=resnet18,
        float_accuracy=0.6976,
        model_params={'pretrained': True},
        evaluation_function=pytorch_model_accuracy_evaluation,
        name="resnet18"
    ),
    'resnet50': ModelParameters(
        model=resnet50,
        float_accuracy=0.8015,
        model_params={'pretrained': True},
        evaluation_function=pytorch_model_accuracy_evaluation,
        interpolation="bicubic",
        name="resnet50"
    ),
    'regnetx_006': ModelParameters(
        model=regnetx_006,
        float_accuracy=0.7386,
        model_params={'pretrained': True},
        evaluation_function=pytorch_model_accuracy_evaluation,
        interpolation="bicubic",
        name="regnetx_006"
    ),
    'regnetx_032': ModelParameters(
        model=regnetx_032,
        float_accuracy=0.78164,
        model_params={'pretrained': True},
        evaluation_function=pytorch_model_accuracy_evaluation,
        interpolation="bicubic",
        name="regnetx_032"
    ),
    'mbv2': ModelParameters(
        model=timm.create_model("mobilenetv2_100", pretrained=True),
        float_accuracy=0.7289,
        model_params={'pretrained': True},
        evaluation_function=pytorch_model_accuracy_evaluation,
        interpolation="bicubic",
        name="mobilenetv2_100"
    ),
}
