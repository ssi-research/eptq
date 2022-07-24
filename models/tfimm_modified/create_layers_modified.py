from typing import Optional, Union, Tuple, List

import tensorflow as tf

from tfimm.layers.conv import get_padding


def create_conv2d(
        kernel_size: Union[int, Tuple[int, int], List],
        nb_experts: Optional[int] = None,
        nb_groups: int = 1,
        depthwise: bool = False,
        **kwargs,
):
    """
    Selects a 2D convolution implementation based on arguments. Creates and returns one
    of Conv2D, DepthwiseConv2D.

    Used extensively by EfficientNet, MobileNetV3 and related networks.
    """
    # We change the default value for use_bias to False.
    kwargs["use_bias"] = kwargs.get("use_bias", False)

    if isinstance(kernel_size, list):
        assert nb_experts is None  # MixNet + CondConv combo not supported currently
        assert nb_groups is None  # MixedConv groups are defined by kernel list
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify
        # non-square kernels
        raise NotImplementedError("MixedConv2D not implemented yet...")
        # m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        if nb_experts is not None:
            raise NotImplementedError("ConvConv2D not implemented yet...")
            # m = CondConv2d(
            #     in_channels, out_channels, kernel_size, groups=groups, **kwargs
            # )
        elif depthwise:
            # Depthwise convolution
            conv = generate_pad_dw_conv(
                kernel_size=kernel_size,
                # depthwise_initializer=FanoutInitializer(depthwise=True),
                **kwargs,
            )
        else:
            # Regular (group-wise) convolution
            conv = generate_pad_conv(
                kernel_size=kernel_size,
                groups=nb_groups,
                # kernel_initializer=FanoutInitializer(nb_groups=nb_groups),
                **kwargs,
            )
    return conv


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return tf.keras.layers.Activation("linear")  # identity layer
    return aa_layer(stride) if issubclass(aa_layer, tf.keras.layers.AveragePooling2D) \
        else aa_layer(channels=channels, stride=stride)


# def create_attn(attn_type, channels, **kwargs):
#     module_cls = get_attn(attn_type)
#     if module_cls is not None:
#         # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
#         return module_cls(channels, **kwargs)
#     return None


def generate_pad_dw_conv(kernel_size,
                         strides=(1, 1),
                         padding="valid",
                         depth_multiplier=1,
                         data_format=None,
                         dilation_rate=(1, 1),
                         groups=1,
                         activation=None,
                         use_bias=True,
                         depthwise_initializer="glorot_uniform",
                         bias_initializer="zeros",
                         depthwise_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         depthwise_constraint=None,
                         bias_constraint=None,
                         **kwargs):
    pad = tf.keras.layers.ZeroPadding2D(
        padding=get_padding(kernel_size, strides, dilation_rate)) if padding == "symmetric" else None

    conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding if padding != "symmetric" else "valid",
                                           depth_multiplier=depth_multiplier,
                                           data_format=data_format,
                                           dilation_rate=dilation_rate,
                                           groups=groups,
                                           activation=activation,
                                           use_bias=use_bias,
                                           depthwise_initializer=depthwise_initializer,
                                           bias_initializer=bias_initializer,
                                           depthwise_regularizer=depthwise_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           depthwise_constraint=depthwise_constraint,
                                           bias_constraint=bias_constraint,
                                           **kwargs)

    def func(x):
        if pad is not None:
            x = pad(x)
        return conv(x)

    return func


def generate_pad_conv(filters,
                      kernel_size,
                      strides=(1, 1),
                      padding="valid",
                      data_format=None,
                      dilation_rate=(1, 1),
                      groups=1,
                      activation=None,
                      use_bias=True,
                      kernel_initializer="glorot_uniform",
                      bias_initializer="zeros",
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None,
                      bias_constraint=None,
                      **kwargs):
    pad = tf.keras.layers.ZeroPadding2D(
        padding=get_padding(kernel_size, strides, dilation_rate)) if padding == "symmetric" else None

    conv = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding if padding != "symmetric" else "valid",
                                  data_format=data_format,
                                  dilation_rate=dilation_rate,
                                  groups=groups,
                                  activation=activation,
                                  use_bias=use_bias,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint,
                                  **kwargs)

    def func(x):
        if pad is not None:
            x = pad(x)
        return conv(x)

    return func
