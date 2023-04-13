"""
 This file is copied from https://github.com/martinsbruveris/tensorflow-image-models
 and modified for this project needs.

 The Licence of the tensorflow-image-models project is shown in: https://github.com/martinsbruveris/tensorflow-image-models/blob/main/LICENSE
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tfimm.layers import (
    norm_layer_factory,
)
from tfimm.models import ModelConfig

# Model registry will add each entrypoint function to this
__all__ = ["MLPMixer", "MLPMixerConfig"]

from models.tfimm_modified.layers.drop import DropPath
from models.tfimm_modified.layers.transformers import PatchEmbeddings, MLP


@dataclass
class MLPMixerConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    embed_dim: int = 512
    nb_blocks: int = 16
    mlp_ratio: Tuple[float, float] = (0.5, 4.0)
    block_layer: str = "mixer_block"
    mlp_layer: str = "mlp"
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    init_values: float = 1e-4  # Initialisation for ResBlocks
    nlhb: bool = False  # Negative logarithmic head bias
    stem_norm: bool = False
    # Parameters for inference
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    # Weight transfer
    first_conv: str = "stem/proj"
    classifier: str = "head"

    @property
    def nb_patches(self) -> int:
        return (self.input_size[0] // self.patch_size) * (
            self.input_size[1] // self.patch_size
        )


class MMixerBlock(object):
    """
    Residual Block w/ token mixing and channel MLPs
    Based on: "MLP-Mixer: An all-MLP Architecture for Vision"
    """

    def __init__(self, cfg: MLPMixerConfig, **kwargs):
        self.name = kwargs['name']

        self.cfg = cfg

        norm_layer = norm_layer_factory(cfg.norm_layer)
        mlp_layer = MLP
        tokens_dim, channels_dim = [int(x * cfg.embed_dim) for x in cfg.mlp_ratio]

        self.norm1 = norm_layer(name=f"{self.name}.norm1")
        self.mlp_tokens = mlp_layer(
            hidden_dim=tokens_dim,
            embed_dim=cfg.nb_patches,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name=f"{self.name}.mlp_tokens",
        )
        self.drop_path = DropPath(drop_prob=cfg.drop_path_rate)
        self.norm2 = norm_layer(name=f"{self.name}.norm2")
        self.mlp_channels = mlp_layer(
            hidden_dim=channels_dim,
            embed_dim=cfg.embed_dim,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name=f"{self.name}.mlp_channels",
        )

    def __call__(self, x: tf.Tensor):
        shortcut = x
        x = self.norm1(x)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.mlp_tokens(x)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.drop_path(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = self.drop_path(x)
        x = x + shortcut

        return x


def generate_mlpmixer_net_keras(cfg: MLPMixerConfig, *args, **kwargs):

    # self.cfg = cfg
    # self.nb_features = cfg.embed_dim

    norm_layer = norm_layer_factory(cfg.norm_layer)
    block_layer = MMixerBlock

    stem = PatchEmbeddings(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        norm_layer=cfg.norm_layer if cfg.stem_norm else "",
        name="stem",
    )
    blocks = [
        block_layer(cfg=cfg, name=f"blocks/{j}") for j in range(cfg.nb_blocks)
    ]
    norm = norm_layer(name="norm")
    head = (
        tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
        if cfg.nb_classes > 0
        else tf.keras.layers.Activation("linear")
    )

    ##########################################
    # Forward
    ##########################################
    x_in = tf.keras.Input([*cfg.input_size, 3], name="input")
    x = stem(x_in)

    for j, block in enumerate(blocks):
        x = block(x)

    x = norm(x)
    x = tf.reduce_mean(x, axis=1)

    x = head(x)

    return tf.keras.Model(inputs=x_in, outputs=x)


def mixer_b16_224():
    """Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    net_name = "mixer_b16_224"
    cfg = MLPMixerConfig(
        name=net_name,
        url="[timm]" + net_name,
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
    )

    return generate_mlpmixer_net_keras, cfg
