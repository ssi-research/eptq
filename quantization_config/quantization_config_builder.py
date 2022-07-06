from model_compression_toolkit import CoreConfig, QuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2


def core_config_builder(mixed_precision, num_calibration_iter, num_samples_for_distance, use_grad_based_weights):
    # TODO: Need to edit the config or is default config is enough?
    quant_config = QuantizationConfig()
    mp_config = None
    if mixed_precision:
        # TODO: set distance_fn and after changing default in library
        mp_config = MixedPrecisionQuantizationConfigV2(num_of_images=num_samples_for_distance,
                                                       use_grad_based_weights=use_grad_based_weights)
    core_config = CoreConfig(num_calibration_iter,
                             quantization_config=quant_config,
                             mixed_precision_config=mp_config)
    return core_config
