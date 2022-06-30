from keras.applications.mobilenet_v2 import MobileNetV2

from datasets.image_utils import get_default_keras_data_preprocess
from model_configs.model_config import TFModelConfig

model_dictionary = {
    'mobilenet_v2': TFModelConfig(
        model=MobileNetV2,
        float_accuracy=0.7185,
        model_params={'weights': 'imagenet'},
)
}
