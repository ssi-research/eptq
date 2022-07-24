from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
from functools import partial

import tensorflow as tf

from tfimm.layers import DropPath, act_layer_factory, norm_layer_factory

from tfimm.models import ModelConfig


from models.tfimm_modified.create_layers_modified import create_conv2d
from models.tfimm_modified.resnet.resnet_blocks import ResNetConv1Block, StemPoolBlock, ReplacedStemPoolBlock, \
    ResNetBasicBlock, DownSampleAvg, DownSampleConv, StemPoolWithAABlock

# Model registry will add each entrypoint fn to this
__all__ = ['ResNet18Config']


@dataclass
class ResNet18Config(ModelConfig):

    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)  # TODO: check input size

    # Architecture
    stem_type: str = ''  # TODO: set default stem type
    stem_width: int = 64
    replace_stem_pool: bool = False
    output_stride: int = 32
    global_pool: str = 'avg'
    cardinality: int = 1
    base_width: int = 64
    block_reduce_first: int = 1
    down_kernel_size: int = 1
    avg_down: bool = False
    zero_init_last: bool = True
    block_args: Dict[str, Any] = field(default_factory=lambda: {'attn_layer': 'se'})  # TODO: block argument, are there any others that we need to set?

    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.
    drop_block_rate: float = 0.

    # Other params
    norm_layer: str = "batch_norm"
    act_layer: str = "relu"
    aa_layer: tf.keras.layers.Layer = None  # TODO: check type and default value of aa_layer
    block_layers: tuple = (2, 2, 2, 2)  # number of layers in each stage block


def generate_resnet18_net_keras(cfg: ResNet18Config, **kwargs):
    norm_layer = norm_layer_factory(cfg.norm_layer)
    act_layer = act_layer_factory(cfg.act_layer)

    # Stem
    deep_stem = 'deep' in cfg.stem_type
    inplanes = cfg.stem_width * 2 if deep_stem else 64  # conv out_channels

    if deep_stem:
        stem_chs = (cfg.stem_width, cfg.stem_width)
        if 'tiered' in cfg.stem_type:
            stem_chs = (3 * (cfg.stem_width // 4), cfg.stem_width)
        conv1 = ResNetConv1Block(cfg.norm_layer, cfg.act_layer, stem_chs, inplanes)

    else:
        conv1 = create_conv2d(
            filters=inplanes,
            kernel_size=7,
            strides=2,
            # padding=3,
            padding='symmetric',
            use_bias=False,
            name="conv1"
        )

    bn1 = norm_layer(name="bn1")
    act1 = act_layer(name="act1")

    # Stem pooling
    if cfg.replace_stem_pool:
        stem_pool = ReplacedStemPoolBlock(cfg.norm_layer, cfg.act_layer, cfg.aa_layer, inplanes)
    else:
        if cfg.aa_layer is not None:
            if issubclass(cfg.aa_layer, tf.keras.layers.AveragePooling2D):
                stem_pool = cfg.aa_layer(2)
            else:
                stem_pool = StemPoolWithAABlock(cfg.aa_layer, inplanes)
        else:
            stem_pool = StemPoolBlock(inplanes)

    # Feature Blocks
    channels = [64, 128, 256, 512]
    stage_block = ResNetBasicBlock
    stage_modules = _make_blocks(
        block_fn=stage_block, channels=channels, block_repeats=cfg.block_layers, inplanes=inplanes, cardinality=cfg.cardinality,
        base_width=cfg.base_width, output_stride=cfg.output_stride, reduce_first=cfg.block_reduce_first,
        avg_down=cfg.avg_down, down_kernel_size=cfg.down_kernel_size, act_layer=cfg.act_layer, norm_layer=cfg.norm_layer,
        aa_layer=cfg.aa_layer, drop_block_rate=cfg.drop_block_rate, drop_path_rate=cfg.drop_path_rate, **cfg.block_args)

    # Head (Pooling and Classifier)
    num_features = 512 * stage_block.expansion  # TODO: where should we use this?

    global_pool = tf.keras.layers.GlobalAveragePooling2D()
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
    x = conv1(x_in)
    x = bn1(x)
    x = act1(x)
    x = stem_pool(x)

    for block_name, blocks in stage_modules:
        for block in blocks:
            print(block_name)
            x = block(x)

    x = global_pool(x)
    x = flatten(x)

    x = drop(x)
    x = classifier(x)
    return tf.keras.Model(inputs=x_in, outputs=x)


def _make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    # feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1

    # drop_blocks = [None, None,
    #                partial(DropBlock2d, drop_prob=drop_block_rate, block_size=5, gamma_scale=0.25) if drop_block_rate else None,
    #                partial(DropBlock2d, drop_prob=drop_block_rate, block_size=3, gamma_scale=1.00) if drop_block_rate else None]
    # TODO: don't need DropBlock2d for ResNet18 because drop_block_rate is 0.
    #  but if needed for other networks then it needs to be implemented (or imported from tfimm if exists there)
    drop_blocks = [None, None, None, None]

    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks)):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = DownSampleAvg(**down_kwargs) if avg_down else DownSampleConv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(
                block_fn(inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                         drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs)
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, blocks))
        # stages.append((stage_name, nn.Sequential(*blocks)))
        # feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    # return stages, feature_info
    return stages


def resnet18():
    """ResNet18 Model"""
    # TODO: set name and url (and other arguments?)
    cfg = ResNet18Config()
    return generate_resnet18_net_keras, cfg


