"""
 This file is copied from https://github.com/martinsbruveris/tensorflow-image-models
 and modified for this project needs.

 The Licence of the tensorflow-image-models project is shown in: https://github.com/martinsbruveris/tensorflow-image-models/blob/main/LICENSE
"""

import tensorflow as tf


class ClassifierHead(object):
    """Classifier head with configurable global pooling and dropout."""

    def __init__(
        self,
        nb_classes: int,
        pool_type: str = "avg",
        drop_rate: float = 0.0,
        use_conv: bool = False,
        **kwargs,
    ):

        pool_type = pool_type or ""  # Convert other falsy values to empty string
        if not pool_type and not use_conv and nb_classes > 0:
            raise ValueError(
                "Pooling can only be disabled if conv classifier is used "
                "or classifier is disabled."
            )
        if pool_type == "":
            self.pool = None
        elif pool_type == "avg":
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
        elif pool_type == "max":
            self.pool = tf.keras.layers.GlobalMaxPool2D()
        else:
            raise NotImplementedError(f"pool_type={pool_type} not implemented.")

        if drop_rate > 0.0:
            self.drop = tf.keras.layers.Dropout(rate=drop_rate)
        else:
            self.drop = None

        if nb_classes == 0:
            self.fc = None
        elif use_conv:
            self.fc = tf.keras.layers.Conv2D(
                filters=nb_classes,
                kernel_size=1,
                use_bias=True,
                name="fc",
            )
        else:
            self.fc = tf.keras.layers.Dense(
                units=nb_classes,
                use_bias=True,
                name="fc",
            )
        if pool_type == "":
            self.flatten = None
        else:
            self.flatten = tf.keras.layers.Flatten()

    def __call__(self, x: tf.Tensor):
        if self.pool is not None:
            x = self.pool(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.fc is not None:
            x = self.fc(x)
        if self.flatten is not None:
            x = self.flatten(x)
        return x
