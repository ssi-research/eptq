"""
 This file is copied from https://github.com/martinsbruveris/tensorflow-image-models
 and modified for this project needs.

 The Licence of the tensorflow-image-models project is shown in: https://github.com/martinsbruveris/tensorflow-image-models/blob/main/LICENSE
"""

import tensorflow as tf


class DropPath(object):
    """
    Per sample stochastic depth when applied in main path of residual blocks
    This is the same as the DropConnect created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in
    a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    Following the usage in `timm` we've changed the name to `drop_path`.
    """

    def __init__(self, drop_prob=None, **kwargs):
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - self.drop_prob

    def __call__(self, x: tf.Tensor):
        if not self.drop_prob > 0.0:
            return x

        # Compute drop_connect tensor
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = self.keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)

        # Rescale output to preserve batch statistics
        x = tf.math.divide(x, self.keep_prob) * binary_tensor
        return x
