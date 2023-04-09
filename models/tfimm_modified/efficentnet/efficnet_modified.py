"""
We provide an implementation and pretrained weights for the EfficientNet family of
models.

Paper: EfficientNet: Rethinking Model Scaling for CNNs.
`[arXiv:1905.11946] <https://arxiv.org/abs/1905.11946>`_.

This code and weights have been ported from the
`timm <https://github.com/rwightman/pytorch-image-models>`_ implementation. It does mean
that some model weights have undergone the journey from TF (original weights from the
Google Brain team) to PyTorch (timm library) back to TF (tfimm port of timm).

The following models are available.

* MobileNet-V2 models. These models correspond to the ``mobilenetv2_...`` models in
  timm.

  * ``mobilenet_v2_{050, 100, 140}``. These are MobileNet-V2 models with channel
    multiplier set to 0.5, 1.0 and 1.4 respectively.
  * ``mobilenet_v2_{110d, 120d}``. These are MobileNet-V2 models with (channel, depth)
    multipliers set to (1.1, 1.2) and (1.2, 1.4) respectively.

* Original EfficientNet models. These models correspond to the models ``tf_...`` in
  timm.

  * ``efficientnet_{b0, b1, b2, b3, b4, b5, b6, b7, b8}``

* EfficientNet AdvProp models, trained with adversarial examples. These models
  correspond to the ``tf_...`` models in timm.

  * ``efficientnet_{b0, ..., b8}_ap``

* EfficientNet NoisyStudent models, trained via semi-supervised learning. These models
  correspond to the ``tf_...`` models in timm.

  * ``efficientnet_{b0, ..., b7}_ns``
  * ``efficientnet_l2_ns_475``
  * ``efficientnet_l2``

* PyTorch versions of the EfficientNet models. These models use symmetric padding rather
  than "same" padding that is default in TF. They correspond to the ``efficientnet_...``
  models in timm.

  * ``pt_efficientnet_{b0, ..., b4}``

* EfficientNet-EdgeTPU models, optimized for inference on Google's Edge TPU hardware.
  These models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_es``
  * ``efficientnet_em``
  * ``efficientnet_el``

* EfficientNet-Lite models, optimized for inference on mobile devices, CPUs and GPUs.
  These models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_lite0``
  * ``efficientnet_lite1``
  * ``efficientnet_lite2``
  * ``efficientnet_lite3``
  * ``efficientnet_lite4``

* EfficientNet-V2 models. These models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_v2_b0``
  * ``efficientnet_v2_b1``
  * ``efficientnet_v2_b2``
  * ``efficientnet_v2_b3``
  * ``efficientnet_v2_s``
  * ``efficientnet_v2_m``
  * ``efficientnet_v2_l``

* EfficientNet-V2 models, pretrained on ImageNet-21k, fine-tuned on ImageNet-1k. These
  models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_v2_s_in21ft1k``
  * ``efficientnet_v2_m_in21ft1k``
  * ``efficientnet_v2_l_in21ft1k``
  * ``efficientnet_v2_xl_in21ft1k``

* EfficientNet-V2 models, pretrained on ImageNet-21k. These models correspond to the
  ``tf_...`` models in timm.

  * ``efficientnet_v2_s_in21k``
  * ``efficientnet_v2_m_in21k``
  * ``efficientnet_v2_l_in21k``
  * ``efficientnet_v2_xl_in21k``
"""
# Hacked together by / Copyright 2019, Ross Wightman
# Copyright 2022 Marting Bruveris
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import tensorflow as tf

from tfimm.architectures.efficientnet_builder import (
    decode_architecture,
    round_channels,
)
from models.tfimm_modified.efficentnet.efficientnet_builder import EfficientNetBuilder
from tfimm.layers import act_layer_factory, norm_layer_factory
from tfimm.models import ModelConfig
from tfimm.utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)

from models.tfimm_modified.efficentnet.efficientnet_blocks import create_conv2d

# Model registry will add each entrypoint fn to this
__all__ = ["EfficientNetConfig"]


# TODO: Fix list_timm_models with two different model names.


@dataclass
class EfficientNetConfig(ModelConfig):
    """
    Configuration class for EfficientNet models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        nb_classes: Number of classes for classification head.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        stem_size: Number of filters in first convolution.
        architecture: Tuple of tuple of strings defining the architecture of residual
            blocks. The outer tuple defines the stages while the inner tuple defines
            the blocks per stage.
        channel_multiplier: Multiplier for channel scaling. One of the three dimensions
            of EfficientNet scaling.
        depth_multiplier: Multiplier for depth scaling. One of the three dimensions of
            EfficientNet scaling.
        fix_first_last:  Fix first and last block depths when multiplier is applied.
        nb_features: Number of features before the classifier layer.

        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.
        padding: Type of padding to use for convolutional layers. Can be one of
            "same", "valid" or "symmetric" (PyTorch-style symmetric padding).

        crop_pct: Crop percentage for ImageNet evaluation.
        interpolation: Interpolation method for ImageNet evaluation.
        mean: Defines preprocessing function. If ``x`` is an image with pixel values
            in (0, 1), the preprocessing function is ``(x - mean) / std``.
        std: Defines preprpocessing function.

        first_conv: Name of first convolutional layer. Used by
            :func:`~tfimm.create_model` to adapt the number in input channels when
            loading pretrained weights.
        classifier: Name of classifier layer. Used by :func:`~tfimm.create_model` to
            adapt the classifier when loading pretrained weights.
    """

    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    # Architecture
    stem_size: int = 32
    architecture: Tuple[Tuple[str, ...], ...] = ()
    channel_multiplier: float = 1.0
    depth_multiplier: float = 1.0
    fix_first_last: bool = False
    nb_features: int = 1280
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other params
    norm_layer: str = "batch_norm"
    act_layer: str = "swish"
    padding: str = "symmetric"  # One of "symmetric", "same", "valid"
    # Parameters for inference
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "conv_stem"
    classifier: str = "classifier"


def generate_efficient_net_keras(cfg: EfficientNetConfig, **kwargs):
    norm_layer = norm_layer_factory(cfg.norm_layer)
    act_layer = act_layer_factory(cfg.act_layer)

    # Stem
    conv_stem = create_conv2d(
        filters=cfg.stem_size,
        kernel_size=3,
        strides=2,
        padding=cfg.padding,
        name="conv_stem",
    )
    bn1 = norm_layer(name="bn1")
    act1 = act_layer()

    # Middle stages (IR/ER/DS Blocks)
    builder = EfficientNetBuilder(
        output_stride=32,
        channel_multiplier=cfg.channel_multiplier,
        padding=cfg.padding,
        act_layer=cfg.act_layer,
        norm_layer=cfg.norm_layer,
        drop_path_rate=cfg.drop_path_rate,
    )
    architecture = decode_architecture(
        architecture=cfg.architecture,
        depth_multiplier=cfg.depth_multiplier,
        depth_truncation="ceil",
        experts_multiplier=1,
        fix_first_last=cfg.fix_first_last,
        group_size=None,
    )
    blocks = builder(architecture)

    # Head
    conv_head = create_conv2d(
        filters=cfg.nb_features,
        kernel_size=1,
        padding=cfg.padding,
        name="conv_head",
    )
    bn2 = norm_layer(name="bn2")
    act2 = act_layer()

    # Pooling + Classifier
    pool = tf.keras.layers.GlobalAveragePooling2D()
    flatten = tf.keras.layers.Flatten()
    drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)
    classifier = (
        tf.keras.layers.Dense(units=cfg.nb_classes, name="classifier")
        if cfg.nb_classes > 0
        else tf.keras.layers.Activation("linear")  # Identity layer
    )
    ##########################################
    # Forward
    ##########################################
    x_in = tf.keras.Input([*cfg.input_size, 3], name="input")
    x = conv_stem(x_in)
    x = bn1(x)
    x = act1(x)

    for key, block in blocks.items():
        x = block(x)

    x = conv_head(x)
    x = bn2(x)
    x = act2(x)

    x = pool(x)
    x = flatten(x)

    x = drop(x)
    x = classifier(x)
    return tf.keras.Model(inputs=x_in, outputs=x)


def _mobilenet_v2_cfg(
        name: str,
        timm_name: str,
        channel_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        fix_stem_head: bool = False,
        crop_pct: float = 0.875,
        mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
):
    """
    Creates the config for a MobileNet-v2 model.

    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py  # noqa: E501
    Paper: https://arxiv.org/abs/1801.04381

    Args:
        name: Model name
        timm_name: Name of model in TIMM
        channel_multiplier: Multiplier for channel scaling.
        depth_multiplier: Multiplier for depth scaling.
        fix_stem_head: Scale stem channels and number of features or not
        crop_pct: Crop percentage for ImageNet evaluation
        mean: Defines preprocessing function.
        std: Defines preprpocessing function.
    """
    round_channels_fn = partial(round_channels, multiplier=channel_multiplier)
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        stem_size=32 if fix_stem_head else round_channels_fn(32),
        architecture=(
            ("ds_r1_k3_s1_c16",),
            ("ir_r2_k3_s2_e6_c24",),
            ("ir_r3_k3_s2_e6_c32",),
            ("ir_r4_k3_s2_e6_c64",),
            ("ir_r3_k3_s1_e6_c96",),
            ("ir_r3_k3_s2_e6_c160",),
            ("ir_r1_k3_s1_e6_c320",),
        ),
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        fix_first_last=fix_stem_head,
        nb_features=1280 if fix_stem_head else max(1280, round_channels_fn(1280)),
        norm_layer="batch_norm",
        act_layer="relu6",
        crop_pct=crop_pct,
        mean=mean,
        std=std,
    )
    return cfg


def mobilenet_v2_100_m():
    """MobileNet-V2 with 1.0 channel multiplier"""
    cfg = _mobilenet_v2_cfg(
        name="mobilenet_v2_100_m",
        timm_name="mobilenetv2_100",
        channel_multiplier=1.0,
    )
    return generate_efficient_net_keras, cfg
