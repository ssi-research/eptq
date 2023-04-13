"""
 This file is copied from https://github.com/martinsbruveris/tensorflow-image-models
 and modified for this project needs.

 The Licence of the tensorflow-image-models project is shown in: https://github.com/martinsbruveris/tensorflow-image-models/blob/main/LICENSE
"""

from dataclasses import dataclass
from typing import Optional, Union, Callable

import numpy as np
import tensorflow as tf


@dataclass
class RegNetCfg:
    depth: int = 21
    w0: int = 80
    wa: float = 42.63
    wm: float = 2.66
    group_size: int = 24
    bottle_ratio: float = 1.
    stem_width: int = 32
    downsample: Optional[str] = 'conv1x1'
    linear_out: bool = False
    num_features: int = 0
    act_layer: Union[str, Callable] = tf.keras.layers.ReLU
    input_size: tuple = (3, 224, 224)
    url: str = '[timm]'


model_cfgs = dict(
    # RegNet-X
    regnetx_006=RegNetCfg(url='[timm]regnetx_006', w0=48, wa=36.97, wm=2.24, group_size=24, depth=16, input_size=(3, 224, 224)),
    regnetx_032=RegNetCfg(url='[timm]regnetx_032', w0=88, wa=26.31, wm=2.25, group_size=48, depth=25, input_size=(3, 224, 224)),
)


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, group_size, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    groups = np.array([group_size for _ in range(num_stages)])
    return widths.astype(int).tolist(), num_stages, groups.astype(int).tolist()


def get_stage_args(cfg: RegNetCfg, default_stride=2, output_stride=32):
    # Generate RegNet ws per block
    widths, num_stages, stage_gs = generate_regnet(cfg.wa, cfg.w0, cfg.wm, cfg.depth, cfg.group_size)

    # Convert to per stage format
    stage_widths, stage_depths = np.unique(widths, return_counts=True)
    stage_br = [cfg.bottle_ratio for _ in range(num_stages)]
    stage_strides = []
    stage_dilations = []
    net_stride = 2
    dilation = 1
    for _ in range(num_stages):
        if net_stride >= output_stride:
            dilation *= default_stride
            stride = 1
        else:
            stride = default_stride
            net_stride *= stride
        stage_strides.append(stride)
        stage_dilations.append(dilation)

    # Adjust the compatibility of ws and gws
    stage_widths, stage_gs = adjust_widths_groups_comp(stage_widths, stage_br, stage_gs)
    arg_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_size']
    per_stage_args = [
        dict(zip(arg_names, params)) for params in
        zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_br, stage_gs)]
    common_args = dict(
        downsample=cfg.downsample, linear_out=cfg.linear_out,
        act_layer=cfg.act_layer)
    return per_stage_args, common_args


def conv_bn_act(x, out_chs, kernel_size=1, stride=1, dilation=1, groups=1, act_layer=None, padding='same', bias=False, name=''):
    x = tf.keras.layers.Conv2D(out_chs, kernel_size, stride, dilation_rate=dilation, groups=groups,
                               padding=padding, use_bias=bias, name=name+'.conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=name+'.bn')(x)
    if act_layer is not None:
        x = act_layer()(x)
    return x


def downsample_conv(x, out_chs, kernel_size=1, stride=1, dilation=1, name=''):
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    return conv_bn_act(x, out_chs, kernel_size, stride, dilation=dilation, act_layer=None, name=name + '.downsample')


def create_shortcut(x,
        downsample_type, in_chs, out_chs, kernel_size, stride, dilation=(1, 1), name=''):
    assert downsample_type in ('conv1x1', '', None)
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        dargs = dict(stride=stride, dilation=dilation[0])
        if not downsample_type:
            return None  # no shortcut, no downsample
        else:
            return downsample_conv(x, out_chs, kernel_size=kernel_size, name=name, **dargs)
    else:
        return x


def bottleneck(x, in_chs, out_chs, stride=1, dilation=(1, 1), bottle_ratio=1, group_size=1,
            downsample='conv1x1', linear_out=False, act_layer=tf.keras.layers.ReLU, name=''):
    bottleneck_chs = int(round(out_chs * bottle_ratio))
    groups = bottleneck_chs // group_size
    shortcut = x

    x = conv_bn_act(x, bottleneck_chs, kernel_size=1, act_layer=act_layer, name=name + '.conv1')

    if stride > 1:
        x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        padding = 'valid'
    else:
        padding = 'same'

    x = conv_bn_act(x, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation[0], groups=groups,
                    act_layer=act_layer, padding=padding, name=name + '.conv2')

    x = conv_bn_act(x, out_chs, kernel_size=1, act_layer=None, name=name + '.conv3')

    x = x + create_shortcut(shortcut, downsample, in_chs, out_chs, kernel_size=1, stride=stride, dilation=dilation, name=name)

    if linear_out or act_layer is None:
        return x
    return act_layer()(x)


def reg_stage(x, depth, in_chs, out_chs, stride, dilation,
            name='', block_fn=bottleneck, **block_kwargs):
    first_dilation = 1 if dilation in (1, 2) else 2
    for i in range(depth):
        block_stride = stride if i == 0 else 1
        block_in_chs = in_chs if i == 0 else out_chs
        block_dilation = (first_dilation, dilation)
        block_name = name + ".b{}".format(i + 1)
        x= block_fn(
                x, block_in_chs, out_chs, stride=block_stride, dilation=block_dilation,
                name=block_name, **block_kwargs)
        first_dilation = dilation
    return x


def build_regnet(cfg: RegNetCfg,  num_classes=1000, output_stride=32):
    assert output_stride in (8, 16, 32)

    # Construct the stem
    stem_width = cfg.stem_width
    act_layer = cfg.act_layer
    feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]
    x_in = tf.keras.Input([*cfg.input_size[1:3], 3], name="input")
    x = tf.keras.layers.ZeroPadding2D(padding=1)(x_in)
    stem = conv_bn_act(x, stem_width, 3, stride=2, act_layer=act_layer, name='stem', padding='valid')

    # Construct the stages
    prev_width = stem_width
    curr_stride = 2
    per_stage_args, common_args = get_stage_args(cfg, output_stride=output_stride)
    assert len(per_stage_args) == 4
    x = stem
    out = []
    for i, stage_args in enumerate(per_stage_args):
        stage_name = "s{}".format(i + 1)
        x = reg_stage(x, in_chs=prev_width, block_fn=bottleneck, **stage_args, **common_args, name=stage_name)
        prev_width = stage_args['out_chs']
        curr_stride *= stage_args['stride']
        feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]
        out.append(x)

    # Construct the head
    if cfg.num_features:
        final_conv = conv_bn_act(x, cfg.num_features, kernel_size=1, act_layer=act_layer)
    else:
        final_conv = cfg.act_layer()(x) if cfg.linear_out else x

    x = tf.keras.layers.AveragePooling2D(7)(final_conv)
    x = tf.keras.layers.Dense(num_classes, name='head.fc')(x)

    return tf.keras.Model(inputs=x_in, outputs=x)


def regnetx_006():
    return build_regnet, model_cfgs['regnetx_006']
