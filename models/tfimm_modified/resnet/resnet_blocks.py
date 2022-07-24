from typing import Tuple

import tensorflow as tf
from tfimm.layers import act_layer_factory, norm_layer_factory
from tfimm.layers.conv import get_padding
from tfimm.layers.attention import attn_layer_factory

from models.tfimm_modified.create_layers_modified import create_aa


class ResNetConv1Block(tf.keras.layers.Layer):
    def __init__(
            self,
            norm_layer: str,
            act_layer: str,
            stem_channels: Tuple[int, int],
            filters: int,
            **kwargs):
        super().__init__(**kwargs)

        self.norm_layer = norm_layer_factory(norm_layer)
        self.act_layer = act_layer_factory(act_layer)
        self.stem_channels = stem_channels
        self.filters = filters

        self.pad1 = tf.keras.layers.ZeroPadding2D(1)
        self.block_conv1 = None
        self.block_bn1 = None
        self.block_act1 = None
        self.pad2 = tf.keras.layers.ZeroPadding2D(1)
        self.block_conv2 = None
        self.block_bn2 = None
        self.block_act2 = None
        self.pad3 = tf.keras.layers.ZeroPadding2D(1)
        self.block_conv3 = None

    def build(self, input_shape):

        self.block_conv1 = tf.keras.layers.Conv2D(
            filters=self.stem_channels[0],
            kernel_size=3,
            strides=2,
            padding='valid',
            # padding=1,
            use_bias=False,
            name="block_conv1",
        )

        self.block_bn1 = self.norm_layer(name="block_bn1")
        self.block_act1 = self.act_layer(name="block_act1")

        self.block_conv2 = tf.keras.layers.Conv2D(
            filters=self.stem_channels[1],
            kernel_size=3,
            strides=1,
            padding='valid',
            # padding=1,
            use_bias=False,
            name="block_conv2",
        )

        self.block_bn2 = self.norm_layer(name="block_bn2")
        self.block_act2 = self.act_layer(name="block_act2")

        self.block_conv3 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding='valid',
            # padding=1,
            use_bias=False,
            name="block_conv3",
        )

    def call(self, x):
        x = self.pad1(x)
        x = self.block_conv1(x)
        x = self.block_bn1(x)
        x = self.block_act1(x)
        x = self.pad2(x)
        x = self.block_conv2(x)
        x = self.block_bn2(x)
        x = self.block_act2(x)
        x = self.pad3(x)
        x = self.block_conv3(x)
        return x


class ReplacedStemPoolBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            norm_layer: str,
            act_layer: str,
            aa_layer: str,
            filters: int,
            **kwargs):
        super().__init__(**kwargs)

        self.aa_layer = aa_layer
        self.filters = filters
        self.norm_layer = norm_layer_factory(norm_layer)
        self.act_layer = act_layer_factory(act_layer)

        self.pad = tf.keras.layers.ZeroPadding2D(padding=1)
        self.pool_conv = None
        self.pool_aa = None
        self.pool_bn = None
        self.pool_act = None

    def build(self, input_shape):
        self.pool_conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1 if self.aa_layer else 2,
            padding='valid',
            use_bias=False,
            name="pool_conv",
        )

        self.pool_aa = create_aa(self.aa_layer, channels=self.filters, stride=2)
        self.pool_bn = self.norm_layer(name="pool_bn")
        self.pool_act = self.act_layer(name="pool_act")

    def call(self, x):
        x = self.pad(x)
        x = self.pool_conv(x)
        x = self.pool_aa(x)
        x = self.pool_bn(x)
        x = self.pool_act(x)
        return x


class StemPoolWithAABlock(tf.keras.layers.Layer):
    def __init__(
            self,
            aa_layer: str,
            filters: int,
            **kwargs):
        super().__init__(**kwargs)

        self.aa_layer = aa_layer
        self.filters = filters

        self.max_pool = None
        self.pool_aa = None
        self.pad = tf.keras.layers.ZeroPadding2D(padding=1)

    def build(self, input_shape):
        # self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), stride=1, padding=1),
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='valid'),
        self.pool_aa = create_aa(self.aa_layer, channels=self.filters, stride=2)

    def call(self, x):
        x = self.pad(x)
        x = self.max_pool(x)
        x = self.pool_aa(x)

        return x


class StemPoolBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            **kwargs):
        super().__init__(**kwargs)

        self.filters = filters

        self.max_pool = None
        self.pool_aa = None
        self.pad = tf.keras.layers.ZeroPadding2D(padding=1)

    def build(self, input_shape):
        # self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), stride=12, padding=1),
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

    def call(self, x):
        x = self.pad(x)
        x = self.max_pool(x)

        return x


class ResNetBasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride, downsample, cardinality, base_width,
            reduce_first, dilation, first_dilation, act_layer, norm_layer,
            attn_layer, aa_layer, drop_block, drop_path, **kwargs):
        super().__init__(**kwargs)

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        self.cardinality = cardinality
        self.base_width = base_width

        self.norm_layer = norm_layer_factory(norm_layer)
        self.act_layer = act_layer_factory(act_layer)
        self.aa_layer = aa_layer
        self.attn_layer = attn_layer_factory(attn_layer)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample

        self.reduce_first = reduce_first
        self.dilation = dilation
        self.first_dilation = first_dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

        # Layers
        self.pad1 = None
        self.conv1 = None
        self.bn1 = None
        self.drop_block = None
        self.act1 = None
        self.aa = None
        self.pad2 = None
        self.conv2 = None
        self.bn2 = None
        self.se = None
        self.act2 = None

    def build(self, input_shape):
        first_planes = self.planes // self.reduce_first
        outplanes = self.planes * self.expansion
        first_dilation = self.first_dilation or self.dilation
        use_aa = self.aa_layer is not None and (self.stride == 2 or first_dilation != self.dilation)

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=first_dilation)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=3,
            strides=1 if self.aa_layer else self.stride,
            # padding=first_dilation,
            padding='valid',
            dilation_rate=(first_dilation, first_dilation),
            use_bias=False,
            name="conv1",
        )

        self.bn1 = self.norm_layer(name="bn1")
        self.drop_block = self.drop_block() if self.drop_block is not None else tf.keras.layers.Activation("linear")  # Identity layer

        self.act1 = self.act_layer(name="act1")
        self.aa = create_aa(self.aa_layer, channels=first_planes, stride=self.stride, enable=use_aa)

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=self.dilation)

        self.conv2 = tf.keras.layers.Conv2D(
            filters=outplanes,
            kernel_size=3,
            strides=1 if self.aa_layer else self.stride,
            # padding=self.dilation,
            padding='valid',
            dilation_rate=(self.dilation, self.dilation),
            use_bias=False,
            name="conv2",
        )

        self.bn2 = self.norm_layer(name="bn2")

        # self.se = create_attn(self.attn_layer, outplanes)
        # self.se = self.attn_layer(outplanes)
        self.se = self.attn_layer()

        self.act2 = self.act_layer(name="act2")

    def call(self, x):
        shortcut = x

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act2(x)

        return x


class DownSampleAvg(tf.keras.layers.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            first_dilation: int,
            norm_layer: str,
            **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg_stride = stride if dilation == 1 else 1
        self.dilation = dilation
        self.first_dilation = first_dilation
        self.norm_layer = norm_layer_factory(norm_layer)

        self.pool = None
        self.conv = None
        self.norm = None

    def build(self, input_shape):
        if self.stride == 1 and self.dilation == 1:
            self.pool = tf.keras.layers.Activation("linear")  # Identity layer
        else:
            if self.avg_stride == 1 and self.dilation > 1:
                self.pool = tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=self.avg_stride,
                    # padding=0,
                    padding='valid',
                    # TODO: how to set these arguments in Keras AveragePooling layer?
                    # ceil_mode=True, count_include_pad=False
                )
            else:
                self.pool = tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=self.avg_stride,
                    # padding=0
                    padding='valid'
                    # TODO: how to set these arguments in Keras AveragePooling layer?
                    # ceil_mode=True, count_include_pad=False
                )

        self.conv = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            strides=1,
            # padding=0,
            padding='valid',
            use_bias=False,
            name="conv",
        )

        self.norm = self.norm_layer(name="norm")

    def call(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)

        return x


class DownSampleConv(tf.keras.layers.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            first_dilation: int,
            norm_layer: str,
            **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
        self.stride = stride
        self.dilation = (first_dilation or dilation) if kernel_size > 1 else 1
        self.first_dilation = first_dilation
        self.norm_layer = norm_layer_factory(norm_layer)

        self.pad = None
        self.conv = None
        self.norm = None

    def build(self, input_shape):
        p = get_padding(self.kernel_size, self.stride, self.first_dilation)

        self.pad = tf.keras.layers.ZeroPadding2D(padding=p)

        self.conv = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='valid',
            dilation_rate=(self.dilation, self.dilation),
            use_bias=False,
            name="conv",
        )

        self.norm = self.norm_layer(name="norm")

    def call(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)

        return x



