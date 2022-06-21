from functools import partial

import keras


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