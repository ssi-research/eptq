import model_compression_toolkit as mct
from constants import DEFAULT_QUANT_BITWIDTH
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, TargetPlatformModel

tp = mct.target_platform


def get_fixed_bitwidth_tp_model(weights_n_bits=DEFAULT_QUANT_BITWIDTH,
                                activation_n_bits=DEFAULT_QUANT_BITWIDTH,
                                enable_weights_quantization=True,
                                enable_activation_quantization=True,
                                is_symmetric: bool = False,
                                is_symmetric_act: bool = False) -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.

    Returns: A TargetPlatformModel object.

    """
    weights_quantization_method = tp.QuantizationMethod.SYMMETRIC if is_symmetric else tp.QuantizationMethod.UNIFORM
    activation_quantization_method = tp.QuantizationMethod.SYMMETRIC if is_symmetric_act else tp.QuantizationMethod.UNIFORM
    default_config = tp.OpQuantizationConfig(
        activation_quantization_method=activation_quantization_method,
        weights_quantization_method=weights_quantization_method,
        weights_n_bits=weights_n_bits,
        activation_n_bits=activation_n_bits,
        weights_per_channel_threshold=True,
        enable_weights_quantization=enable_weights_quantization,
        enable_activation_quantization=enable_activation_quantization,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None)

    no_quant_config = tp.OpQuantizationConfig(
        activation_quantization_method=activation_quantization_method,
        weights_quantization_method=weights_quantization_method,
        weights_n_bits=weights_n_bits,  # does not affect quantization
        activation_n_bits=activation_n_bits,  # does not affect quantization
        weights_per_channel_threshold=True,
        enable_weights_quantization=False,
        enable_activation_quantization=False,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        weights_multiplier_nbits=None)

    return generate_tp_model(default_config=default_config,
                             no_quant_config=no_quant_config)


def generate_tp_model(default_config: OpQuantizationConfig,
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
    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = tp.QuantizationConfigOptions([default_config])

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    generated_tp_model = tp.TargetPlatformModel(default_configuration_options, name='fixed_bitwidth_tp_model')

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the TargetPlatformModel instance, and create them as below:
    with generated_tp_model:
        # Create an OperatorsSet to represent a set of operations.
        # Each OperatorsSet has a unique label.
        # If a quantization configuration options is passed, these options will
        # be used for operations that will be attached to this set's label.
        # Otherwise, it will be a configure-less set (used in fusing):

        # Configure OperatorsSet for layers that we don't want to quantize.
        tp.OperatorsSet("NoQuantization", tp.QuantizationConfigOptions([no_quant_config]))

        # Configure OperatorsSet for layers that have parameters that we might want to quantize.
        conv = tp.OperatorsSet("Conv")
        fc = tp.OperatorsSet("FullyConnected")

        # Configure OperatorsSet for layers that have only activation that we might want to quantize.
        # Separated operators' definition can be useful for creating fusing patterns.
        any_relu = tp.OperatorsSet("AnyReLU")
        add = tp.OperatorsSet("Add")
        sub = tp.OperatorsSet("Sub")
        mul = tp.OperatorsSet("Mul")
        div = tp.OperatorsSet("Div")
        prelu = tp.OperatorsSet("PReLU")
        swish = tp.OperatorsSet("Swish")
        sigmoid = tp.OperatorsSet("Sigmoid")
        tanh = tp.OperatorsSet("Tanh")
        tp.OperatorsSet("Input")

        # Combine multiple operators into a single operator to avoid quantization between
        # them. To do this we define fusing patterns using the OperatorsSets that were created.
        # To group multiple sets with regard to fusing, an OperatorSetConcat can be created
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
