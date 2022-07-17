""" EfficientNet, MobileNetV3, etc Builder

Assembles EfficieNet and related network feature blocks from string definitions.
Handles stride, dilation calculations, and selects feature extraction points.

Hacked together by / Copyright 2019, Ross Wightman
"""

from collections import OrderedDict
from typing import List

from models.tfimm_modified.efficentnet.efficientnet_blocks import (
    BlockArgs,
    ConvBnAct,
    # DepthwiseSeparableConv,
    MDepthwiseSeparableConv,
    EdgeResidual,
    # InvertedResidual,
    MInvertedResidual
)

from tfimm.architectures.efficientnet_builder import round_channels, _log

_DEBUG_BUILDER = False


class EfficientNetBuilder:
    """
    Build Trunk Blocks

    Adapted from timm.
    """

    def __init__(
            self,
            output_stride=32,
            channel_multiplier: float = 1.0,
            padding="",
            se_from_exp=False,  # ???
            act_layer=None,
            norm_layer=None,
            drop_path_rate=0.0,
    ):
        self.output_stride = output_stride
        self.channel_multiplier = channel_multiplier
        self.padding = padding
        # Calculate se channel reduction from expanded (mid) chs
        self.se_from_exp = se_from_exp
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.drop_path_rate = drop_path_rate

    def _make_block(
            self,
            block_args: BlockArgs,
            stage_idx: int,
            block_idx: int,
            total_idx: int,
            nb_blocks: int,
    ):
        block_name = f"blocks.{stage_idx}.{block_idx}"
        # Stochastic depth
        drop_path_rate = self.drop_path_rate * total_idx / nb_blocks

        block_type = block_args.block_type
        block_args.filters = round_channels(block_args.filters, self.channel_multiplier)
        if block_args.force_in_channels is not None:
            block_args.force_in_channels = round_channels(
                block_args.force_in_channels, self.channel_multiplier
            )
        block_args.padding = self.padding
        block_args.norm_layer = self.norm_layer
        # block act fn overrides the model default
        block_args.act_layer = block_args.act_layer or self.act_layer
        assert block_args.act_layer is not None

        block_args.drop_path_rate = drop_path_rate
        if block_type != "cn":
            # TODO: Add parameter se_from_exp (used in Mobilenet v3) which does not
            #       adjust se_ratio.
            block_args.se_ratio /= block_args.exp_ratio

        if block_type == "ir":
            if block_args.nb_experts is not None:
                # TODO: Not implemented yet
                _log(f"  CondConvResidual {block_idx}, Args: {str(block_args)}")
                block = CondConvResidual(cfg=block_args, name=block_name)  # noqa: F821
            else:
                _log(f"  InvertedResidual {block_idx}, Args: {str(block_args)}")
                block = MInvertedResidual(cfg=block_args, name=block_name)
        elif block_type in {"ds", "dsa"}:
            _log(f"  DepthwiseSeparable {block_idx}, Args: {str(block_args)}")
            block = MDepthwiseSeparableConv(cfg=block_args, name=block_name)
        elif block_type == "er":
            _log(f"  EdgeResidual {block_idx}, Args: {str(block_args)}")
            block = EdgeResidual(cfg=block_args, name=block_name)
        elif block_type == "cn":
            _log(f"  ConvBnAct {block_idx}, Args: {str(block_args)}")
            block = ConvBnAct(cfg=block_args, name=block_name)  # noqa: F821
        else:
            raise ValueError(f"Unknown block type {block_type} while building model.")

        return block

    def __call__(self, architecture: List[List[BlockArgs]]) -> OrderedDict:
        """
        Build the blocks

        Args:
            architecture: A list of lists, outer list defines stages, inner list
                contains block configuration(s).

        Return:
             OrderedDict of block layers.
        """
        _log(f"Building model trunk with {len(architecture)} stages...")
        total_block_count = sum([len(x) for x in architecture])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        blocks = OrderedDict()

        # outer list of block_args defines the stacks
        for stack_idx, stack_args in enumerate(architecture):
            _log(f"Stack: {stack_idx}")
            assert isinstance(stack_args, list)

            # Each stack (stage of blocks) contains a list of block arguments
            for block_idx, block_args in enumerate(stack_args):
                _log(f" Block: {block_idx}")

                assert block_args.stride in {1, 2}
                # Only the first block in any stack can have a stride > 1
                if block_idx >= 1:
                    block_args.stride = 1

                next_dilation = current_dilation
                if block_args.stride > 1:
                    next_output_stride = current_stride * block_args.stride
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args.stride
                        block_args.stride = 1
                        _log(
                            f"  Converting stride to dilation to maintain output "
                            f"stride of{self.output_stride}."
                        )
                    else:
                        current_stride = next_output_stride
                block_args.dilation_rate = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                block = self._make_block(
                    block_args,
                    stage_idx=stack_idx,
                    block_idx=block_idx,
                    total_idx=total_block_idx,
                    nb_blocks=total_block_count,
                )
                blocks[f"stage_{stack_idx}/block_{block_idx}"] = block

                total_block_idx += 1  # incr global block idx (across all stacks)
        return blocks
