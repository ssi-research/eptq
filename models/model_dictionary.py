from keras.applications.mobilenet_v2 import MobileNetV2
from models.model_config import ModelParameters
from datasets.image_utils import keras_model_accuracy_evaluation_timm
from models.regnet import regnetx_006, regnetx_032
from models.tfimm_modified.mlp_mixer.mlp_mixer_modified import mixer_b16_224
from models.tfimm_modified.efficentnet.efficnet_modified import mobilenet_v2_100_m
from models.tfimm_modified.resnet.resnet_modified import resnet18, resnet50, tv_resnet50
import tfimm

model_dictionary = {
    'mobilenet_v2': ModelParameters(
        model=MobileNetV2,
        float_accuracy=0.7185,
        model_params={'weights': 'imagenet'},
    ),
    'resnet18': ModelParameters(
        model=resnet18,
        float_accuracy=0.6976,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        name="resnet18"
    ),
    'resnet50': ModelParameters(
        model=resnet50,
        float_accuracy=0.8011,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        interpolation="bicubic"
    ),
    'tv_resnet50': ModelParameters(
        model=tv_resnet50,
        float_accuracy=0.8011,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        interpolation="bicubic"
    ),
    'regnetx_006': ModelParameters(
        model=regnetx_006,
        float_accuracy=0.7386,
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        interpolation="bicubic"
    ),
    'regnetx_032': ModelParameters(
        model=regnetx_032,
        float_accuracy=0.78164,
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        interpolation="bicubic"
    ),
    'mbv2': ModelParameters(
        model=mobilenet_v2_100_m,
        float_accuracy=0.7289,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        interpolation="bicubic"
    ),
    'mlp_mixer': ModelParameters(
        model=mixer_b16_224,
        float_accuracy=0.727,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        name="mixer_b16_224",
        interpolation="bicubic"
    ),
}
