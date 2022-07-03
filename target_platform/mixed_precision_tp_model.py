from typing import List, Tuple

import model_compression_toolkit as mct
from constants import DEFAULT_QUANT_BITWIDTH, DEFAULT_MP_WEIGHTS_OPTIONS, DEFAULT_MP_ACTIVATION_OPTIONS
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, TargetPlatformModel

tp = mct.target_platform


def get_mixed_precision_tp_model(mixed_precision_options,
                                 enable_weights_quantization=True,
                                 enable_activation_quantization=True) -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.

    Returns: A TargetPlatformModel object.

    """
    assert len(mixed_precision_options) > 0, \
        "Mixed-precision search requires at list one mixed precision bit-width option"

    mixed_precision_options.sort(reverse=True)
    max_weights_bitwidth, max_activation_bitwidth = mixed_precision_options[0]

    default_config = tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=max_weights_bitwidth,
        weights_n_bits=max_activation_bitwidth,
        weights_per_channel_threshold=True,
        enable_weights_quantization=enable_weights_quantization,
        enable_activation_quantization=enable_activation_quantization,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None)

    no_quant_config = tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=max_activation_bitwidth,  # does not affect quantization
        weights_n_bits=max_activation_bitwidth,  # does not affect quantization
        weights_per_channel_threshold=True,
        enable_weights_quantization=False,
        enable_activation_quantization=False,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None)

    mp_config_options = [default_config.clone_and_edit(weights_n_bits=weights_n_bits,
                                                       activation_n_bits=activation_n_bits)
                         for weights_n_bits, activation_n_bits in mixed_precision_options]

    return generate_mixed_precision_tp_model(default_config=default_config,
                                             mp_config_options=mp_config_options,
                                             no_quant_config=no_quant_config)


def generate_mixed_precision_tp_model(default_config: OpQuantizationConfig,
                                      mp_config_options: List[OpQuantizationConfig],
                                      no_quant_config: OpQuantizationConfig) -> TargetPlatformModel:
    """
    Generates TargetPlatformModel with default defined Operators Sets, based on the given base configuration and
    mixed-precision configurations options list.

    Args
        default_config: A default OpQuantizationConfig to set as the TP model default configuration.
            In this fixed bit-width TP model all quantized layers will use default_config for quantization configuration.
        no_quant_config: A OpQuantizationConfig to set for layers that we don't want to quantize.

    Returns: A TargetPlatformModel object.

    """
    default_configuration_options = tp.QuantizationConfigOptions([default_config])

    generated_tp_model = tp.TargetPlatformModel(default_configuration_options, name='mixed_precision_tp_model')

    with generated_tp_model:
        tp.OperatorsSet("NoQuantization", tp.QuantizationConfigOptions([no_quant_config]))

        mixed_precision_configuration_options = tp.QuantizationConfigOptions(mp_config_options,
                                                                             base_config=default_config)  # default config is used as based config also

        conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
        fc = tp.OperatorsSet("FullyConnected", mixed_precision_configuration_options)

        any_relu = tp.OperatorsSet("AnyReLU", mixed_precision_configuration_options)
        add = tp.OperatorsSet("Add", mixed_precision_configuration_options)
        sub = tp.OperatorsSet("Sub", mixed_precision_configuration_options)
        mul = tp.OperatorsSet("Mul", mixed_precision_configuration_options)
        div = tp.OperatorsSet("Div", mixed_precision_configuration_options)
        prelu = tp.OperatorsSet("PReLU", mixed_precision_configuration_options)
        swish = tp.OperatorsSet("Swish", mixed_precision_configuration_options)
        sigmoid = tp.OperatorsSet("Sigmoid", mixed_precision_configuration_options)
        tanh = tp.OperatorsSet("Tanh", mixed_precision_configuration_options)

        activations_after_conv_to_fuse = tp.OperatorSetConcat(any_relu, swish, prelu, sigmoid, tanh)
        activations_after_fc_to_fuse = tp.OperatorSetConcat(any_relu, swish, sigmoid)
        any_binary = tp.OperatorSetConcat(add, sub, mul, div)

        # ------------------- #
        # Fusions
        # ------------------- #
        tp.Fusing([conv, activations_after_conv_to_fuse])
        tp.Fusing([fc, activations_after_fc_to_fuse])
        tp.Fusing([conv, add, any_relu])
        tp.Fusing([conv, any_relu, add])
        tp.Fusing([any_binary, any_relu])

    return generated_tp_model
