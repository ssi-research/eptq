import tensorflow as tf
import timm.data
import torch
from tensorflow import keras
from datasets.image_utils import get_default_keras_model_preprocess, keras_model_accuracy_evaluation, \
    get_default_keras_data_preprocess
from models.tfimm_modified.load_weight_updated import load_pytorch_weights_in_tf2_model
from torch.utils.data import RandomSampler, Subset


class ModelParameters(object):
    def __init__(self,
                 model,
                 float_accuracy,
                 preprocess=get_default_keras_model_preprocess(),
                 evaluation_function=keras_model_accuracy_evaluation,
                 model_params={},
                 is_tfimm: bool = False,
                 interpolation="bilinear",
                 image_size=(224, 224, 3),

                 ):
        self.model = model
        self.float_accuracy = float_accuracy
        self.preprocess = preprocess
        self.evaluation_function = evaluation_function
        self.model_params = model_params
        self.is_tfimm = is_tfimm
        self.interpolation = interpolation
        self.image_size = image_size
        self.resize_to = [self.image_size[0], self.image_size[1]]

    def get_float_accuracy(self):
        return self.float_accuracy

    def get_model(self):
        if self.is_tfimm:
            model_fn, cfg = self.model()
            model = model_fn(cfg)

            pt_model = timm.create_model(cfg.url.split("]")[-1], pretrained=True)
            pt_state_dict = pt_model.state_dict()
            load_pytorch_weights_in_tf2_model(model, pt_state_dict)
            # from keras import Input
            # model = tfimm.create_model(self.model, pretrained=True)
            # p = tfimm.create_preprocessing(self.model, dtype="float32")
            #
            # # self.resize_to = [int(s * (1 / 0.875)) for s in self.resize_to]
            #
            # def _preprocess(in_x, in_y):
            #     # _x = tf.image.central_crop(in_x, 0.875)
            #     return p(in_x), in_y
            #
            # self.preprocess = [_preprocess]
            # x = Input(self.image_size, name="input")
            # y = model(x)
            # return tf.keras.Model(inputs=x, outputs=y)
            return model
        if isinstance(self.model, str):
            return tf.keras.models.load_model(self.model, compile=False)
        else:
            return self.model(**self.model_params)

    def get_validation_dataset(self, dir_path, batch_size=50, image_size=(224, 224)):
        if not self.is_tfimm:
            dataset = keras.utils.image_dataset_from_directory(
                directory=dir_path,
                batch_size=batch_size,
                image_size=self.resize_to,
                shuffle=False,
                crop_to_aspect_ratio=True,
                interpolation=self.interpolation)

            for p in self.preprocess:
                dataset = dataset.map(p)

            return dataset
        else:
            ds = timm.data.create_dataset("", dir_path)
            return timm.data.create_loader(ds, image_size, batch_size, use_prefetcher=False)

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
        if not self.is_tfimm:
            dataset = tf.keras.utils.image_dataset_from_directory(
                in_dir,
                batch_size=batch_size,
                image_size=self.resize_to,
                labels=None,
                crop_to_aspect_ratio=True)
            dataset = dataset.take(num_images)

            print(f'Loaded representative dataset. Size: {dataset.cardinality() * batch_size} images')

            for p in preprocessing:
                dataset = dataset.map(p)

            iterator = self.dataset_iterator(dataset)

            def representative_dataset():
                return [next(iterator)]
        else:
            ds = timm.data.create_dataset("", in_dir)
            ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, num_images)))
            iterator = iter(timm.data.create_loader(ds, image_size, batch_size, use_prefetcher=False))

            def representative_dataset():
                x = next(iterator)[0]
                return [torch.permute(x, [0, 2, 3, 1]).cpu().numpy()]

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
