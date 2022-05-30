"""
 This file is copied from https://github.com/martinsbruveris/tensorflow-image-models
 and modified for this project needs.

 The Licence of the tensorflow-image-models project is shown in: https://github.com/martinsbruveris/tensorflow-image-models/blob/main/LICENSE
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import timm

import tensorflow as tf

from tfimm.layers import (
    norm_layer_factory,
)
from tfimm.models import ModelConfig
from tfimm.utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


# model_registry will add each entrypoint fn to this
__all__ = ["ViT", "MViTBlock", "ViTConfig"]

from models.tfimm_modified.layers.drop import DropPath
from models.tfimm_modified.layers.transformers import PatchEmbeddings, MLP


@dataclass
class ViTConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_layer: str = "patch_embeddings"
    patch_nb_blocks: tuple = ()
    patch_size: int = 16
    embed_dim: int = 768
    nb_blocks: int = 12
    nb_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    representation_size: Optional[int] = None
    distilled: bool = False
    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    # Parameters for inference
    interpolate_input: bool = False
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = IMAGENET_INCEPTION_MEAN
    std: Tuple[float, float, float] = IMAGENET_INCEPTION_STD
    first_conv: str = "patch_embed/proj"
    # DeiT models have two classifier heads, one for distillation
    classifier: Union[str, Tuple[str, str]] = "head"
    pt_state_dict: dict = None

    """
    Args:
        nb_classes: Number of classes for classification head
        in_channels: Number of input channels
        input_size: Input image size
        patch_layer: Layer used to transform image to patches. Possible values are
            `patch_embeddings` and `hybrid_embeddings`.
        patch_nb_blocks: When `patch_layer="hybrid_embeddings`, this is the number of
            residual blocks in each stage. Set to `()` to use only the stem.
        patch_size: Patch size; Image size must be multiple of patch size. For hybrid
            embedding layer, this patch size is applied after the convolutional layers.
        embed_dim: Embedding dimension
        nb_blocks: Depth of transformer (number of encoder blocks)
        nb_heads: Number of self-attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Enable bias for qkv if True
        representation_size: Enable and set representation layer (pre-logits) to this
            value if set
        distilled: Model includes a distillation token and head as in DeiT models
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Dropout rate for stochastic depth
        norm_layer: Normalization layer
        act_layer: Activation function
    """

    @property
    def nb_tokens(self) -> int:
        """Number of special tokens"""
        return 2 if self.distilled else 1

    @property
    def grid_size(self) -> Tuple[int, int]:
        grid_size = (
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        if self.patch_layer == "hybrid_embeddings":
            # 2 reductions in the stem, 1 reduction in each stage except the first one
            reductions = 2 + max(len(self.patch_nb_blocks) - 1, 0)
            stride = 2 ** reductions
            grid_size = (grid_size[0] // stride, grid_size[1] // stride)
        return grid_size

    @property
    def nb_patches(self) -> int:
        """Number of patches without class and distillation tokens."""
        return self.grid_size[0] * self.grid_size[1]


class MViTMultiHeadAttention(object):
    def __init__(
            self,
            embed_dim: int,
            nb_heads: int,
            qkv_bias: bool,
            drop_rate: float,
            attn_drop_rate: float,
            **kwargs,
    ):
        self.name = kwargs['name']

        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        head_dim = embed_dim // nb_heads
        self.scale = head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(
            units=3 * embed_dim, use_bias=qkv_bias, name=f"{self.name}.qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=embed_dim, name=f"{self.name}.proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=drop_rate)

    def __call__(self, x: tf.Tensor):

        # B (batch size), N (sequence length), D (embedding dimension),
        # H (number of heads)
        seq_length = tf.shape(x)[1]._inferred_value[0]
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = tf.keras.layers.Reshape(target_shape=(seq_length, 3, self.nb_heads, -1))(qkv)
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.scale * tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, N, N)
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, N, N)
        attn = self.attn_drop(attn)

        x = tf.linalg.matmul(attn, v)  # (B, H, N, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, H, D/H)
        x = tf.keras.layers.Reshape(target_shape=(seq_length, -1))(x)  # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MViTBlock(object):
    def __init__(
            self,
            embed_dim: int,
            nb_heads: int,
            mlp_ratio: float,
            qkv_bias: bool,
            drop_rate: float,
            attn_drop_rate: float,
            drop_path_rate: float,
            norm_layer: str,
            act_layer: str,
            **kwargs,
    ):
        self.name = kwargs['name']

        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        norm_layer = norm_layer_factory(norm_layer)

        self.norm1 = norm_layer(name=f"{self.name}.norm1")
        self.attn = MViTMultiHeadAttention(
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            name=f"{self.name}.attn",
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = norm_layer(name=f"{self.name}.norm2")
        self.mlp = MLP(
            hidden_dim=int(embed_dim * mlp_ratio),
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            act_layer=act_layer,
            name=f"{self.name}.mlp",
        )

    def __call__(self, x: tf.Tensor):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)

        x = self.drop_path(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + shortcut
        return x


class Token(tf.keras.layers.Layer):
    def __init__(
            self,
            shape: tuple,
            name: str,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.shape = shape
        self.token_name = name

        self.token = None

    def build(self, input_shape):
        self.token = self.add_weight(
            shape=self.shape,
            initializer="zeros",
            trainable=True,
            name=self.token_name,
        )

    def call(self, x, training=False, return_features=False):
        return self.token

    def get_config(self):
        return {
            'shape': self.shape,
            'name': self.token_name
        }


def generate_deit_net_keras(cfg: ViTConfig, *args, **kwargs):
    nb_features = cfg.embed_dim  # For consistency with other models
    norm_layer = norm_layer_factory(cfg.norm_layer)
    cfg = cfg

    if cfg.patch_layer == "patch_embeddings":
        patch_embed = PatchEmbeddings(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            norm_layer="",  # ViT does not use normalization in patch embeddings
            name="patch_embed",
        )
    else:
        raise ValueError(f"Unknown patch layer: {cfg.patch_layer}.")

    pos_drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)

    blocks = [
        MViTBlock(
            embed_dim=cfg.embed_dim,
            nb_heads=cfg.nb_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=cfg.norm_layer,
            act_layer=cfg.act_layer,
            name=f"blocks/{j}",
        )
        for j in range(cfg.nb_blocks)
    ]
    norm = norm_layer(name="norm")

    # Some models have a representation layer on top of cls token

    cls_token_layer = tf.keras.layers.Dense(
        cfg.embed_dim,
        kernel_initializer='zeros',
        use_bias=True,
        name="cls_token",
    )

    dist_token_layer = tf.keras.layers.Dense(
        cfg.embed_dim,
        kernel_initializer='zeros',
        use_bias=True,
        name="dist_token",
    )

    pos_embed_layer = tf.keras.layers.Dense(
        cfg.embed_dim * (cfg.nb_patches + cfg.nb_tokens),
        kernel_initializer='zeros',
        use_bias=True,
        name="pos_embed",
    )

    # Classifier head(s)
    head = (
        tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
        if cfg.nb_classes > 0
        else tf.keras.layers.Activation("linear")  # Identity layer
    )

    if cfg.distilled:
        head_dist = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head_dist")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )
    else:
        head_dist = None

    ##########################################
    # Forward
    ##########################################
    x_in = tf.keras.Input([*cfg.input_size, 3], name="input")

    x = patch_embed(x_in)

    x_dummy = x[:, 0:1, :]
    x_cls = cls_token_layer(x_dummy)
    x_dist = dist_token_layer(x_dummy)
    x_embed = pos_embed_layer(x_dummy)

    if not cfg.distilled:
        x = tf.keras.layers.Concatenate(axis=1)([x_cls, x])
    else:
        x = tf.keras.layers.Concatenate(axis=1)([x_cls, x_dist, x])

    x_embed = tf.keras.layers.Reshape(x.shape.as_list()[1:])(x_embed)
    x = x + x_embed

    x = pos_drop(x)

    for j, block in enumerate(blocks):
        x = block(x)

    x = norm(x)

    if cfg.distilled:
        # Here we diverge from timm and return both outputs as one tensor. That way
        # all models always have one output by default
        x = x[:, :2]
    else:
        x = x[:, 0]

    if not cfg.distilled:
        x = head(x)
    else:
        y = head(x[:, 0])
        y_dist = head_dist(x[:, 1])

        y_avg = (y + y_dist) / 2

        x = tf.nn.softmax(y_avg)

    model = tf.keras.Model(inputs=x_in, outputs=x)
    return model


def deit_base_distilled_patch16_224():
    """
    DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    net_name = "deit_base_distilled_patch16_224"
    url = "[timm]" + net_name
    pt_model = timm.create_model(url.split("]")[-1], pretrained=True)
    pt_state_dict = pt_model.state_dict()

    cfg = ViTConfig(
        name=net_name,
        url=url,
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        distilled=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
        pt_state_dict=pt_state_dict
    )
    return generate_deit_net_keras, cfg
