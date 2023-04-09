from models.model_config import ModelParameters
from datasets.image_utils import keras_model_accuracy_evaluation_timm
from models.tfimm_modified.mlp_mixer.mlp_mixer_modified import mixer_b16_224
from models.tfimm_modified.efficentnet.efficnet_modified import mobilenet_v2_100_m
from models.tfimm_modified.regnet.regnet_modified import regnetx_006
from models.tfimm_modified.resnet.resnet_modified import resnet18, resnet50
from models.tfimm_modified.vit.vit_modified import deit_base_distilled_patch16_224

model_dictionary = {
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
    'regnetx_006': ModelParameters(
        model=regnetx_006,
        float_accuracy=0.7386,
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
    'deit': ModelParameters(
        model=deit_base_distilled_patch16_224,
        float_accuracy=83.336,
        model_params={'weights': 'imagenet'},
        is_tfimm=True,
        evaluation_function=keras_model_accuracy_evaluation_timm,
        name="deit_base_distilled_patch16_224",
        interpolation="bicubic",
        allow_missing_keys=True,
    ),
}
