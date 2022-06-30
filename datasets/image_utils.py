import os
from functools import partial
import keras
from constants import TRAIN_DIR


def get_imagenet_folders_by_class(dir=TRAIN_DIR, class_numbers=None):
    class_folders = [os.path.join(dir, f) for f in os.listdir(dir)]
    if isinstance(class_numbers, int):
        return [class_folders[class_numbers]]
    elif isinstance(class_numbers, list):
        return [class_folder for class_index, class_folder in enumerate(class_folders) if class_index in class_numbers]
    else:
        return class_folders


def get_default_keras_model_preprocess():

    def imagenet_preprocess_input(images, labels):
        return keras.applications.imagenet_utils.preprocess_input(images, mode='tf'), labels

    return [imagenet_preprocess_input]


def get_default_keras_data_preprocess(args):
    return [partial(keras.preprocessing.image.smart_resize, size=(args.image_size, args.image_size)),
            partial(keras.applications.imagenet_utils.preprocess_input, mode='tf')]


def keras_model_accuracy_evaluation(model, dataset):
    model.compile(metrics=['accuracy'])
    result = model.evaluate(dataset)
    return result[1]