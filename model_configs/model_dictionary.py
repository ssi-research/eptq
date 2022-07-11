from keras.applications.mobilenet_v2 import MobileNetV2
from model_configs.model_config import ModelParameters

model_dictionary = {
    'mobilenet_v2': ModelParameters(
        model=MobileNetV2,
        float_accuracy=0.7185,
        model_params={'weights': 'imagenet'},
    ),
    'resnet18': ModelParameters(
        model="resnet18",
        float_accuracy=0.7185,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
    ),
    'mbv2': ModelParameters(
        model="mobilenet_v2_100",
        float_accuracy=0.7185,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
    )
}
