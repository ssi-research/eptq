from functools import partial
import keras
import torch
import numpy as np
from tqdm import tqdm


def get_default_keras_model_preprocess():
    def imagenet_preprocess_input(images, labels):
        return keras.applications.imagenet_utils.preprocess_input(images, mode='tf'), labels

    return [imagenet_preprocess_input]


def get_default_keras_data_preprocess(image_size):
    return [partial(keras.preprocessing.image.smart_resize, size=(image_size, image_size)),
            partial(keras.applications.imagenet_utils.preprocess_input, mode='tf')]


def keras_model_accuracy_evaluation(model, dataset):
    model.compile(metrics=['accuracy'])
    result = model.evaluate(dataset)
    return result[1]


def keras_model_accuracy_evaluation_timm(model, dataset):
    model.compile(metrics=['accuracy'])
    total = 0
    count = 0
    for x, y in tqdm(dataset):
        x = torch.permute(x, [0, 2, 3, 1]).cpu().numpy()
        y_hat = model.predict(x)
        y = y.cpu().numpy()

        count += np.sum(np.argmax(y_hat, axis=-1) == y)
        total += x.shape[0]
    return count / total
