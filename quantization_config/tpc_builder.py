from quantization_config.target_platform.fix_bitwidth_tp_model import get_fixed_bitwidth_tp_model
from quantization_config.target_platform.mixed_precision_tp_model import get_mixed_precision_tp_model
from utils.mixed_precision_utils import MP_BITWIDTH_OPTIONS_DICT, MPCONFIG

MP_NAME = 'mixed_precision_tpc'
FIXED_NAME = 'fixed_bitwidth_tpc'


def build_target_platform_capabilities(generate_fw_tpc, mixed_precision: bool, activation_nbits: int, weights_nbits: int,
                                       disable_weights_quantization: bool,
                                       disable_activation_quantization: bool, weights_cr, activation_cr, total_cr,
                                       mixed_precision_config: MPCONFIG = MPCONFIG.MP_PARTIAL_CANDIDATES,
                                       is_symmetric: bool = False,
                                       is_symmetric_act: bool = False):
    # TODO: Add logging
    bit_width_mapping = MP_BITWIDTH_OPTIONS_DICT[mixed_precision_config]
    if mixed_precision:
        weights_mp = weights_cr is not None or total_cr is not None
        activation_mp = activation_cr is not None or total_cr is not None
        activation_bits = bit_width_mapping if activation_mp else [activation_nbits]
        weights_bits = bit_width_mapping if weights_mp else [weights_nbits]

        mixed_precision_options = [(w, a) for w in weights_bits for a in activation_bits]
        target_platform_model = get_mixed_precision_tp_model(mixed_precision_options=mixed_precision_options,
                                                             enable_weights_quantization=not disable_weights_quantization,
                                                             enable_activation_quantization=not disable_activation_quantization,
                                                             is_symmetric=is_symmetric,
                                                             is_symmetric_act=is_symmetric_act)
        target_platform_cap = generate_fw_tpc(target_platform_model, name=MP_NAME)
    else:
        # TODO: maybe create a dictionary of TP models
        # TODO: allow to get config from input (bitwidth and enable quantization)?
        target_platform_model = get_fixed_bitwidth_tp_model(weights_n_bits=weights_nbits,
                                                            activation_n_bits=activation_nbits,
                                                            enable_weights_quantization=not disable_weights_quantization,
                                                            enable_activation_quantization=not disable_activation_quantization,
                                                            is_symmetric=is_symmetric,
                                                            is_symmetric_act=is_symmetric_act)
        target_platform_cap = generate_fw_tpc(target_platform_model, name=FIXED_NAME)
    return target_platform_cap, bit_width_mapping
