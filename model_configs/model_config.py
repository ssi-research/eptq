from abc import abstractmethod
import tensorflow as tf
from tensorflow import keras

from datasets.image_utils import get_default_keras_preprocess, keras_model_accuracy_evaluation


class ModelConfig(object):
    def __init__(self,
                 model,
                 float_accuracy,
                 preprocess,
                 evaluation_function,
                 framework,
                 model_params={},
                 ):

        self.model = model
        self.float_accuracy = float_accuracy
        self.preprocess = preprocess
        self.evaluation_function = evaluation_function
        self.model_params = model_params

    @abstractmethod
    def get_model(self):
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_model method.')

    @abstractmethod
    def get_data_loader(self):
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_validation_loader method.')

    def get_float_accuracy(self):
        return self.float_accuracy


class TFModelConfig(ModelConfig):
    def __init__(self,
                 model,
                 float_accuracy,
                 preprocess=get_default_keras_preprocess(),
                 evaluation_function=keras_model_accuracy_evaluation,
                 model_params={},
                 ):
        super().__init__(model,
                         float_accuracy,
                         preprocess,
                         evaluation_function,
                         model_params)

    def get_model(self):
        if isinstance(self.model, str):
            return tf.keras.models.load_model(self.model, compile=False)
        else:
            return self.model(**self.model_params)

    def get_dataset(self, dir_path, batch_size=50, image_size=(224, 224), shuffle=False,
                    crop_to_aspect_ratio=True):

        dataset = keras.utils.image_dataset_from_directory(
            directory=dir_path,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            crop_to_aspect_ratio=crop_to_aspect_ratio)

        for p in self.preprocess:
            dataset = dataset.map(p)

        return dataset
