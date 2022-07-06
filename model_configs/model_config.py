import tensorflow as tf
from tensorflow import keras

from datasets.image_utils import get_default_keras_model_preprocess, keras_model_accuracy_evaluation, \
    get_default_keras_data_preprocess


class ModelParameters(object):
    def __init__(self,
                 model,
                 float_accuracy,
                 preprocess=get_default_keras_model_preprocess(),
                 evaluation_function=keras_model_accuracy_evaluation,
                 model_params={},
                 ):
        self.model = model
        self.float_accuracy = float_accuracy
        self.preprocess = preprocess
        self.evaluation_function = evaluation_function
        self.model_params = model_params

    def get_float_accuracy(self):
        return self.float_accuracy

    def get_model(self):
        if isinstance(self.model, str):
            return tf.keras.models.load_model(self.model, compile=False)
        else:
            return self.model(**self.model_params)

    def get_validation_dataset(self, dir_path, batch_size=50, image_size=(224, 224)):

        dataset = keras.utils.image_dataset_from_directory(
            directory=dir_path,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
            crop_to_aspect_ratio=True)

        for p in self.preprocess:
            dataset = dataset.map(p)

        return dataset

    @staticmethod
    def as_numpy_iterator_with_reset(dataset: tf.data.Dataset):
        numpy_iter = dataset.as_numpy_iterator()
        while True:
            for img in numpy_iter:
                yield img
            numpy_iter = dataset.as_numpy_iterator()

    @staticmethod
    def dataset_iterator(dataset: tf.data.Dataset):
        while True:
            for img in dataset:
                yield img

    def get_representative_dataset_from_classes_and_images(self, in_dir, num_images,
                                                           preprocessing, image_size=224, batch_size=1,
                                                           random_seed=0):
        """
            Images returned belong to a set of image_numbers from each class in class numbers.
        class_numbers: The directory of the dataset.
        image_start: The class numbers to use in the dataset.
        num_images_per_class: Number of images to take from each class.
        preprocessing: List of preprocessing functions to perform on the images.
        image_size: Size of each returned image.
        batch_size: The images batch size.
        random_batch: If True, return random ordered images each iteration
        random_seed: Random seed to initialize the shuffling.
        return_numpy_iterator: If True returns images in numpy format. If False returns tf.Tensor (default: False).
        :return: A representative dataset function.
        """
        dataset = tf.keras.utils.image_dataset_from_directory(
            in_dir,
            batch_size=batch_size,
            image_size=(image_size, image_size),
            labels=None,
            crop_to_aspect_ratio=True)
        dataset = dataset.take(num_images)

        print(f'Loaded representative dataset. Size: {dataset.cardinality() * batch_size} images')

        for p in preprocessing:
            dataset = dataset.map(p)

        iterator = self.dataset_iterator(dataset)

        def representative_dataset():
            return [next(iterator)]

        return representative_dataset

    def get_representative_dataset(self, representative_dataset_folder, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
        if preprocessing is None:
            preprocessing = get_default_keras_data_preprocess(image_size)

        return self.get_representative_dataset_from_classes_and_images(
            in_dir=representative_dataset_folder,
            num_images=n_images,
            preprocessing=preprocessing,
            batch_size=batch_size,
            random_seed=seed)
