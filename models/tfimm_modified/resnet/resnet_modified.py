"""
TensorFlow implementation of ResNets
Based on timm/models/resnet.py by Ross Wightman.
It includes the following models:
- Resnets from the PyTorch model hub
- ResNets trained by Ross Wightman
- Models pretrained on weakly-supervised data, finetuned on ImageNet
  Paper: Exploring the Limits of Weakly Supervised Pretraining
  Link: https://arxiv.org/abs/1805.00932
- Models pretrained in semi-supervised way on YFCC100M, finetuned on ImageNet
  Paper: Billion-scale Semi-Supervised Learning for Image Classification
  Link: https://arxiv.org/abs/1905.00546
- ResNets with ECA layers
  Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
  Link: https://arxiv.org/pdf/1910.03151.pdf
- ResNets with anti-aliasing layers
  Paper: Making Convolutional Networks Shift-Invariant Again
  Link: https://arxiv.org/pdf/1904.11486.pdf
- Models from the "Revisiting ResNets" paper
  Paper: Revisiting ResNets
  Link: https://arxiv.org/abs/2103.07579
- ResNets with a squeeze-and-excitation layer
  Paper: Squeeze-and-Excitation Networks
  Link: https://arxiv.org/abs/1709.01507
Copyright 2021 Martins Bruveris
Copyright 2021 Ross Wightman
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf

from tfimm.layers import (
    BlurPool2D,
    act_layer_factory,
    attn_layer_factory,
    norm_layer_factory,
)
from tfimm.models import ModelConfig
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ["ResNetConfig", "MBasicBlock"]

from models.tfimm_modified.layers.classifier import ClassifierHead
from models.tfimm_modified.layers.drop import DropPath


@dataclass
class ResNetConfig(ModelConfig):
    """
    Configuration class for ResNet models.
    """

    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    # Residual blocks
    block: str = "basic_block"
    nb_blocks: Tuple = (2, 2, 2, 2)
    nb_channels: Tuple = (64, 128, 256, 512)
    cardinality: int = 1  # Number of groups in bottleneck conv
    base_width: int = 64  # Determines number of channels in block
    downsample_mode: str = "conv"
    zero_init_last_bn: bool = True
    # Stem
    stem_width: int = 64
    stem_type: str = ""
    replace_stem_pool: bool = False
    # Other params
    block_reduce_first: int = 1
    down_kernel_size: int = 1
    act_layer: str = "relu"
    norm_layer: str = "batch_norm"
    aa_layer: str = ""
    attn_layer: str = ""
    se_ratio: float = 0.0625
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Head
    global_pool: str = "avg"
    # Parameters for inference
    test_input_size: Optional[Tuple[int, int]] = None
    pool_size: int = 7  # For test-time pooling (not implemented yet)
    crop_pct: float = 0.875
    interpolation: str = "bilinear"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "conv1"
    classifier: str = "fc"

    def __post_init__(self):
        if self.test_input_size is None:
            self.test_input_size = self.input_size


class MBasicBlock(object):
    expansion = 1

    def __init__(
            self,
            cfg: ResNetConfig,
            nb_channels: int,
            stride: int,
            drop_path_rate: float,
            downsample_layer,
            **kwargs,
    ):

        assert cfg.cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert cfg.base_width == 64, "BasicBlock does not support changing base width"

        self.name = kwargs['name']

        self.cfg = cfg
        self.downsample_layer = downsample_layer
        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.attn_layer = attn_layer_factory(cfg.attn_layer)

        # Num channels after first conv
        first_planes = nb_channels // cfg.block_reduce_first
        out_planes = nb_channels * self.expansion  # Num channels after second conv
        use_aa = cfg.aa_layer and stride == 2

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=3,
            # If we use anti-aliasing, the anti-aliasing layer takes care of strides
            strides=1 if use_aa else stride,
            use_bias=False,
            name=f"{self.name}.conv1",
        )
        self.bn1 = self.norm_layer(name=f"{self.name}.bn1")
        self.act1 = self.act_layer()
        self.aa = BlurPool2D(stride=stride) if use_aa else None

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            use_bias=False,
            name=f"{self.name}.conv2",
        )
        initializer = "zeros" if cfg.zero_init_last_bn else "ones"
        if cfg.norm_layer == "batch_norm":
            # Only batch norm layer has moving_variance_initializer parameter
            self.bn2 = self.norm_layer(
                gamma_initializer=initializer,
                moving_variance_initializer=initializer,
                name=f"{self.name}.bn2",
            )
        else:
            self.bn2 = self.norm_layer(gamma_initializer=initializer, name=f"{self.name}.bn2")
        if cfg.attn_layer == "se":
            self.se = self.attn_layer(rd_ratio=cfg.se_ratio, name=f"{self.name}.se")
        else:
            self.se = self.attn_layer(name=f"{self.name}.se")
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.act2 = self.act_layer()

    def __call__(self, x: tf.Tensor):
        shortcut = x

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.se(x)

        x = self.drop_path(x)

        if self.downsample_layer is not None:
            shortcut = self.downsample_layer(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class MBottleneck(object):
    expansion = 4

    def __init__(
            self,
            cfg: ResNetConfig,
            nb_channels: int,
            stride: int,
            drop_path_rate: float,
            downsample_layer,
            **kwargs,
    ):
        self.name = kwargs['name']

        self.cfg = cfg
        self.downsample_layer = downsample_layer
        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.attn_layer = attn_layer_factory(cfg.attn_layer)

        # Number of channels after second convolution
        width = int(math.floor(nb_channels * (cfg.base_width / 64)) * cfg.cardinality)
        # Number of channels after first convolution
        first_planes = width // cfg.block_reduce_first
        # Number of channels after third convolution
        out_planes = nb_channels * self.expansion
        use_aa = cfg.aa_layer and stride == 2

        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=1,
            use_bias=False,
            name=f"{self.name}.conv1",
        )
        self.bn1 = self.norm_layer(name=f"{self.name}.bn1")
        self.act1 = self.act_layer()

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=width,
            kernel_size=3,
            # If we use anti-aliasing, the anti-aliasing layer takes care of strides
            strides=1 if use_aa else stride,
            groups=cfg.cardinality,
            use_bias=False,
            name=f"{self.name}.conv2",
        )
        self.bn2 = self.norm_layer(name=f"{self.name}.bn2")
        self.act2 = self.act_layer()
        self.aa = BlurPool2D(stride=stride) if use_aa else None

        self.conv3 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=1,
            use_bias=False,
            name=f"{self.name}.conv3",
        )
        initializer = "zeros" if cfg.zero_init_last_bn else "ones"
        if cfg.norm_layer == "batch_norm":
            # Only batch norm layer has moving_variance_initializer parameter
            self.bn3 = self.norm_layer(
                gamma_initializer=initializer,
                moving_variance_initializer=initializer,
                name=f"{self.name}.bn3",
            )
        else:
            self.bn3 = self.norm_layer(gamma_initializer=initializer, name=f"{self.name}.bn3")
        if cfg.attn_layer == "se":
            self.se = self.attn_layer(rd_ratio=cfg.se_ratio, name=f"{self.name}.se")
        else:
            self.se = self.attn_layer(name=f"{self.name}.se")
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.act3 = self.act_layer()

    def __call__(self, x: tf.Tensor):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.se(x)

        x = self.drop_path(x)

        if self.downsample_layer is not None:
            shortcut = self.downsample_layer(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_avg(cfg: ResNetConfig, out_channels: int, stride: int, name: str):
    norm_layer = norm_layer_factory(cfg.norm_layer)

    if stride != 1:
        pool = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=stride, padding="same"
        )
    else:
        pool = tf.keras.layers.Activation("linear")
    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        name=name + "/downsample/1",
    )
    bn = norm_layer(name=name + "/downsample/2")
    return tf.keras.Sequential([pool, conv, bn])


def downsample_conv(cfg: ResNetConfig, out_channels: int, stride: int, name: str):
    norm_layer = norm_layer_factory(cfg.norm_layer)

    # This layer is part of the conv layer in pytorch and so is not being tracked here
    p = (stride + cfg.down_kernel_size) // 2 - 1
    pad = tf.keras.layers.ZeroPadding2D(padding=p)

    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=cfg.down_kernel_size,
        strides=stride,
        use_bias=False,
        name=name + "/downsample/0",
    )
    bn = norm_layer(name=name + "/downsample/1")
    return tf.keras.Sequential([pad, conv, bn])


def make_stage(
        idx: int,
        in_channels: int,
        cfg: ResNetConfig,
        name: str,
):
    stage_name = f"layer{idx + 1}"  # Weight compatibility requires this name

    assert cfg.block in {"basic_block", "bottleneck"}
    block_cls = MBasicBlock if cfg.block == "basic_block" else MBottleneck
    nb_blocks = cfg.nb_blocks[idx]
    nb_channels = cfg.nb_channels[idx]
    # The actual number of channels after the block. Not the same as nb_channels,
    # because Bottleneck blocks have an expansion factor = 4.
    out_channels = nb_channels * block_cls.expansion

    assert cfg.downsample_mode in {"avg", "conv"}
    downsample_fn = downsample_avg if cfg.downsample_mode == "avg" else downsample_conv

    # We need to know the absolute number of blocks to set stochastic depth decay
    total_nb_blocks = sum(cfg.nb_blocks)
    total_block_idx = sum(cfg.nb_blocks[:idx])

    blocks = []
    for block_idx in range(nb_blocks):
        stride = 1 if idx == 0 or block_idx > 0 else 2
        if (block_idx == 0) and (stride != 1 or in_channels != out_channels):
            downsample_layer = downsample_fn(
                cfg, out_channels, stride, name=f"{stage_name}/0"
            )
        else:
            downsample_layer = None

        # Stochastic depth linear decay rule
        block_dpr = cfg.drop_path_rate * total_block_idx / (total_nb_blocks - 1)

        blocks.append(
            block_cls(
                cfg,
                nb_channels=nb_channels,
                stride=stride,
                downsample_layer=downsample_layer,
                drop_path_rate=block_dpr,
                name=f"{stage_name}/{block_idx}",
            )
        )

        in_channels = nb_channels
        total_block_idx += 1
    return blocks, in_channels


def generate_resnet_net_keras(cfg: ResNetConfig, **kwargs):
    act_layer = act_layer_factory(cfg.act_layer)
    norm_layer = norm_layer_factory(cfg.norm_layer)

    if cfg.stem_type in {"deep", "deep_tiered"}:
        in_channels = cfg.stem_width * 2
        if cfg.stem_type == "deep_tiered":
            stem_chns = (3 * (cfg.stem_width // 4), cfg.stem_width)
        else:
            stem_chns = (cfg.stem_width, cfg.stem_width)

        pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
        conv1_0 = tf.keras.layers.Conv2D(
            filters=stem_chns[0],
            kernel_size=3,
            strides=2,
            use_bias=False,
            name=f"conv1/0",
        )
        bn1_0 = norm_layer(name=f"conv1/1")
        act1_0 = act_layer()
        conv1_1 = tf.keras.layers.Conv2D(
            filters=stem_chns[1],
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"conv1/3",
        )
        bn1_1 = norm_layer(name=f"conv1/4")
        act1_1 = act_layer()
        conv1_2 = tf.keras.layers.Conv2D(
            filters=in_channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"conv1/6",
        )
        conv1 = tf.keras.Sequential(
            [conv1_0, bn1_0, act1_0, conv1_1, bn1_1, act1_1, conv1_2]
        )
    else:
        in_channels = 64
        # In TF "same" padding with strides != 1 is not the same as (3, 3) padding
        # in pytorch, hence the need for an explicit padding layer
        pad1 = tf.keras.layers.ZeroPadding2D(padding=3)
        conv1 = tf.keras.layers.Conv2D(
            filters=in_channels,
            kernel_size=7,
            strides=2,
            use_bias=False,
            name="conv1",
        )
    bn1 = norm_layer(name="bn1")
    act1 = act_layer()

    # Stem Pooling
    if cfg.replace_stem_pool:
        # Note that if replace_stem_pool=True, we are ignoring the aa_layer
        # None of the timm models use both.
        pad = tf.keras.layers.ZeroPadding2D(padding=1)
        conv = tf.keras.layers.Conv2D(
            filters=in_channels,
            kernel_size=3,
            strides=2,
            use_bias=False,
            name=f"maxpool/0",
        )
        bn = norm_layer(name=f"maxpool/1")
        act = act_layer()
        maxpool = tf.keras.Sequential([pad, conv, bn, act])
    else:
        if cfg.aa_layer:
            pad = tf.keras.layers.ZeroPadding2D(padding=1)
            pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1)
            aa = BlurPool2D(stride=2, name=f"maxpool/2")
            maxpool = tf.keras.Sequential([pad, pool, aa])
        else:
            pad = tf.keras.layers.ZeroPadding2D(padding=1)
            pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
            maxpool = tf.keras.Sequential([pad, pool])

    blocks = []
    for idx in range(4):
        stage_blocks, in_channels = make_stage(
            idx=idx, in_channels=in_channels, cfg=cfg, name=''
        )
        blocks.extend(stage_blocks)

    # Head (pooling and classifier)
    head = ClassifierHead(
        nb_classes=cfg.nb_classes,
        pool_type=cfg.global_pool,
        drop_rate=cfg.drop_rate,
        use_conv=False,
        name="",
    )

    ##########################################
    # Forward
    ##########################################
    x_in = tf.keras.Input([*cfg.input_size, 3], name="input")
    x = pad1(x_in)
    x = conv1(x)
    x = bn1(x)
    x = act1(x)
    x = maxpool(x)

    for j, block in enumerate(blocks):
        x = block(x)

    x = head(x)

    return tf.keras.Model(inputs=x_in, outputs=x)


def resnet18():
    """Constructs a ResNet-18 model."""
    net_name = 'resnet18'
    cfg = ResNetConfig(
        name=net_name,
        url="[timm]" + net_name,
        block="basic_block",
        nb_blocks=(2, 2, 2, 2)
    )
    return generate_resnet_net_keras, cfg


def resnet50():
    """Constructs a ResNet-50 model."""
    net_name = 'resnet50'
    cfg = ResNetConfig(
        name=net_name,
        url="[timm]" + net_name,
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        interpolation="bicubic",
        crop_pct=0.95,
    )
    return generate_resnet_net_keras, cfg
