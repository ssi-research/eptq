from quantization_config.target_platform.fix_bitwidth_tp_model import get_fixed_bitwidth_tp_model
from quantization_config.target_platform.mixed_precision_tp_model import get_mixed_precision_tp_model
from quantization_config.target_platform.tensorflow_tpc import generate_keras_tpc
from quantization_config.target_kpi.mixed_precision_utils import MP_BITWIDTH_OPTIONS_DICT, MPCONFIG

MP_NAME = 'mixed_precision_tpc'
FIXED_NAME = 'fixed_bitwidth_tpc'


def build_target_platform_model(mixed_precision: bool, activation_nbits: int, weights_nbits: int,
                                disable_weights_quantization: bool,
                                disable_activation_quantization: bool,
                                mixed_precision_config: MPCONFIG = MPCONFIG.MP_PARTIAL_CANDIDATES):
    # TODO: Add logging
    if mixed_precision:
        mixed_precision_options = [(w, a)
                                   for w in MP_BITWIDTH_OPTIONS_DICT[mixed_precision_config]
                                   for a in MP_BITWIDTH_OPTIONS_DICT[mixed_precision_config]]
        target_platform_model = get_mixed_precision_tp_model(mixed_precision_options=mixed_precision_options,
                                                             enable_weights_quantization=not disable_weights_quantization,
                                                             enable_activation_quantization=not disable_activation_quantization)
        target_platform_cap = generate_keras_tpc(target_platform_model, name=MP_NAME)
    else:
        # TODO: maybe create a dictionary of TP models
        # TODO: allow to get config from input (bitwidth and enable quantization)?
        target_platform_model = get_fixed_bitwidth_tp_model(weights_n_bits=weights_nbits,
                                                            activation_n_bits=activation_nbits,
                                                            enable_weights_quantization=not disable_weights_quantization,
                                                            enable_activation_quantization=not disable_activation_quantization)
        target_platform_cap = generate_keras_tpc(target_platform_model, name=FIXED_NAME)
    return target_platform_cap
