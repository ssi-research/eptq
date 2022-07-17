from keras.applications.mobilenet_v2 import MobileNetV2
from models.model_config import ModelParameters
from datasets.image_utils import keras_model_accuracy_evaluation_timm
from models.tfimm_modified.efficentnet.efficnet_modified import mobilenet_v2_100_m

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
        model=mobilenet_v2_100_m,
        float_accuracy=0.7289,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        # interpolation="bicubic"
    )
}
