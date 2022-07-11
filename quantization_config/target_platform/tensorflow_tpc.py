import tensorflow as tf

from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, \
    Dropout, MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D

import model_compression_toolkit as mct

tp = mct.target_platform


def generate_keras_tpc(tp_model: tp.TargetPlatformModel, name: str):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.

    Args:
        name: Name of the TargetPlatformCapabilities.
        tp_model: TargetPlatformModel object.

    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    keras_tpc = tp.TargetPlatformCapabilities(tp_model, name=name)

    with keras_tpc:
        tp.OperationsSetToLayers("NoQuantization", [Reshape,
                                                    tf.reshape,
                                                    Flatten,
                                                    Cropping2D,
                                                    ZeroPadding2D,
                                                    Dropout,
                                                    MaxPooling2D,
                                                    tf.split,
                                                    tf.quantization.fake_quant_with_min_max_vars,
                                                    tf.math.argmax,
                                                    tf.shape,
                                                    tf.__operators__.getitem,
                                                    tf.compat.v1.shape])

        tp.OperationsSetToLayers("Conv", [Conv2D,
                                          DepthwiseConv2D,
                                          tf.nn.conv2d,
                                          tf.nn.depthwise_conv2d])
        tp.OperationsSetToLayers("FullyConnected", [Dense])
        tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                             tf.nn.relu6,
                                             tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                             tp.LayerFilterParams(Activation, activation="relu")])
        tp.OperationsSetToLayers("Add", [tf.add, Add])
        tp.OperationsSetToLayers("Sub", [tf.subtract, Subtract])
        tp.OperationsSetToLayers("Mul", [tf.math.multiply, Multiply])
        tp.OperationsSetToLayers("Div", [tf.math.divide])
        tp.OperationsSetToLayers("PReLU", [PReLU])
        tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
        tp.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
        tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])

    return keras_tpc