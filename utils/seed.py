import random
import numpy as np
import torch
import tensorflow as tf


def set_seed(seed: int):
    # Generate Numpy
    np.random.seed(seed)
    random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Tensorflow and Keras
    tf.keras.utils.set_random_seed(seed)
