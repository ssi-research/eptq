import numpy as np
import os
from constants import TRAIN_DIR, ALL_CLASSES, N_CLASSES, CLASSES_SUBSET, ONE_CLASS, N_IMAGES, DATASET_TYPES, MAX_IMAGES
from datasets.data_loader import FolderImageLoader
from datasets.image_utils import get_default_keras_data_preprocess


def get_imagenet_folders_by_class(dir=TRAIN_DIR, class_numbers=None):
    class_folders = [os.path.join(dir, f) for f in os.listdir(dir)]
    if isinstance(class_numbers, int):
        return [class_folders[class_numbers]]
    elif isinstance(class_numbers, list):
        return [class_folder for class_index, class_folder in enumerate(class_folders) if class_index in class_numbers]
    else:
        return class_folders


def get_representative_dataset_fn(img_sampler, batch_size):
    def representative_data_gen() -> list:
        sample = []
        for _ in range(batch_size):
            sample.append(next(img_sampler))
        sample = np.concatenate(sample, axis=0)

        return [sample]

    return representative_data_gen


def gen_sample_from_indices(image_data_loaders, image_numbers):
    max_images = max([dl.n_files for dl in image_data_loaders])
    img_ind = 0
    class_ind = 0

    # not all class folders in imagenet have the same number of images
    if len(image_numbers) > max_images:
        image_numbers = image_numbers[:max_images]

    while True:
        image_data_loaders[class_ind].reset(image_numbers[img_ind])
        yield image_data_loaders[class_ind].sample()
        class_ind += 1
        if class_ind == len(image_data_loaders):
            class_ind = 0
            img_ind += 1
            if img_ind == len(image_numbers):
                img_ind = 0


def get_representative_dataset_from_classes_and_images(dir, class_numbers, image_numbers, preprocessing, batch_size=1,
                                                       random_batch=False):
    """
        Images returned belong to a set of image_numbers from each class in class numbers.
    :param class_numbers: The directory of the dataset.
    :param class_numbers: The class numbers to use in the dataset.
    :param image_numbers: The image numbers to take from each class
    :param preprocessing: List of preprocessing functions to perform on the images.
    :param batch_size: The images batch size.
    :return: A representative dataset function.
    """
    class_folders = get_imagenet_folders_by_class(dir, class_numbers)
    image_data_loaders = []
    for class_folder in class_folders:
        image_data_loaders.append(FolderImageLoader(class_folder,
                                                    preprocessing=preprocessing,
                                                    batch_size=1,
                                                    random_batch=random_batch))
    img_sampler = gen_sample_from_indices(image_data_loaders, image_numbers)
    return get_representative_dataset_fn(img_sampler, batch_size)


def get_representative_dataset(args, preprocessing=None):
    if preprocessing is None:
        preprocessing = get_default_keras_data_preprocess()
    if args.dataset_type == ALL_CLASSES:
        # All the images in the imagenet training set
        representative_data_gen = get_representative_dataset_from_classes_and_images(
            dir=args.representative_dataset_folder,
            class_numbers=None,
            image_numbers=list(
                range(args.n_images)),
            preprocessing=preprocessing,
            batch_size=args.batch_size,
            random_batch=args.random_batch)
    elif args.dataset_type == N_CLASSES:
        # Subset of one image per class, for args.n_classes from the 1000 classes in the imagenet training set
        representative_data_gen = get_representative_dataset_from_classes_and_images(
            dir=args.representative_dataset_folder,
            class_numbers=list(range(args.n_classes)), image_numbers=[args.img_num], preprocessing=preprocessing,
            batch_size=args.batch_size, random_batch=args.random_batch)

    elif args.dataset_type == CLASSES_SUBSET:
        # All the images from a subset of args.n_classes from the imagenet training set
        representative_data_gen = get_representative_dataset_from_classes_and_images(
            dir=args.representative_dataset_folder,
            class_numbers=list(range(args.n_classes)),
            image_numbers=[(img_num + args.img_num) % MAX_IMAGES for img_num in range(MAX_IMAGES)],
            preprocessing=preprocessing, batch_size=args.batch_size, random_batch=args.random_batch)

    elif args.dataset_type == ONE_CLASS:
        # All the images from one class, args.class_num, from the imagenet training set
        representative_data_gen = get_representative_dataset_from_classes_and_images(
            dir=args.representative_dataset_folder, class_numbers=args.class_num,
            image_numbers=[(img_num + args.img_num) % MAX_IMAGES for img_num in range(MAX_IMAGES)],
            preprocessing=preprocessing,
            batch_size=args.batch_size, random_batch=args.random_batch)

    elif args.dataset_type == N_IMAGES:
        # args.n_images images from one class, args.class_num, from the imagenet training set
        representative_data_gen = get_representative_dataset_from_classes_and_images(
            dir=args.representative_dataset_folder, class_numbers=args.class_num,
            image_numbers=[(img_num + args.img_num) % args.n_images for img_num in range(args.n_images)],
            preprocessing=preprocessing, batch_size=args.batch_size, random_batch=args.random_batch)
    else:
        raise Exception(
            f'Dataset type \'{args.dataset_type}\' is not a valid choice. Please choose a dataset type out of {DATASET_TYPES}')

    return representative_data_gen
