from abc import abstractmethod
import tensorflow as tf
from tensorflow import keras
from constants import N_CLASSES, ONE_CLASS, N_IMAGES, DATASET_TYPES, MAX_IMAGES, TRAIN_SET
from datasets.image_utils import get_default_keras_model_preprocess, keras_model_accuracy_evaluation, \
    get_imagenet_folders_by_class, get_default_keras_data_preprocess


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

    @abstractmethod
    def get_representative_dataset_from_classes_and_images(self):
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_representative_dataset_from_classes_and_images method.')

    def get_representative_dataset(self, args, preprocessing=None):
        if preprocessing is None:
            preprocessing = get_default_keras_data_preprocess(args)

        if args.dataset_type == TRAIN_SET:
            # All the images in the imagenet training set
            class_numbers = None
            image_start = 0
            num_images_per_class = MAX_IMAGES
        elif args.dataset_type == N_CLASSES:
            # Subset of one image per class, for args.n_classes from the 1000 classes in the imagenet training set
            class_numbers = list(range(args.n_classes))
            image_start = args.img_num
            num_images_per_class = args.n_images
        elif args.dataset_type == ONE_CLASS:
            # All the images from one class, args.class_num, from the imagenet training set
            class_numbers = args.class_num
            image_start = args.img_num
            num_images_per_class = MAX_IMAGES
        elif args.dataset_type == N_IMAGES:
            # args.n_images images from one class, args.class_num, from the imagenet training set
            class_numbers = args.class_num
            image_start = args.img_num
            num_images_per_class = args.n_images
        else:
            raise Exception(
                f'Dataset type \'{args.dataset_type}\' is not a valid choice. Please choose a dataset type out of {DATASET_TYPES}')

        return self.get_representative_dataset_from_classes_and_images(
            dir=args.representative_dataset_folder,
            class_numbers=class_numbers,
            image_start=image_start,
            num_images_per_class=num_images_per_class,
            preprocessing=preprocessing,
            batch_size=args.batch_size,
            random_batch=args.random_batch,
            random_seed=args.random_seed)


class TFModelConfig(ModelConfig):
    def __init__(self,
                 model,
                 float_accuracy,
                 preprocess=get_default_keras_model_preprocess(),
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

    def get_representative_dataset_from_classes_and_images(self, dir, class_numbers, image_start, num_images_per_class,
                                                           preprocessing, image_size=224, batch_size=1,
                                                           random_batch=False,
                                                           random_seed=0, return_numpy_iterator=False):
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
        class_folders = get_imagenet_folders_by_class(dir, class_numbers)
        datasets_list = []
        for class_folder in class_folders:
            ds = tf.keras.utils.image_dataset_from_directory(
                class_folder,
                batch_size=batch_size,
                image_size=(image_size, image_size),
                labels=None,
                crop_to_aspect_ratio=True)

            # circle the dataset so that it starts from "image_start", i.e. if the dataset is range(10)
            # and "image_start=2" than the new dataset is [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
            ds = ds.skip(image_start).concatenate(ds.take(image_start))

            # take only "num_images_per_class" images from the dataset
            ds = ds.take(int(num_images_per_class / batch_size))

            datasets_list.append(ds)

        dataset = datasets_list[0]
        for ds in datasets_list[1:]:
            dataset = dataset.concatenate(ds)

        print(f'Loaded representative dataset. Size: {dataset.cardinality() * batch_size} images')

        for p in preprocessing:
            dataset = dataset.map(p)

        if random_batch:
            dataset = dataset.shuffel(buffer_size=batch_size, seed=random_seed)

        if return_numpy_iterator:
            iterator = self.as_numpy_iterator_with_reset(dataset)
        else:
            iterator = self.dataset_iterator(dataset)

        def representative_dataset():
            return next(iterator)

        return representative_dataset
