from model_compression_toolkit import CoreConfig, QuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
import numpy as np


def compute_mse(float_tensor: np.ndarray, fxp_tensor: np.ndarray, norm: bool = True, norm_eps: float = 1e-8) -> float:
    """
    Compute the mean square error between two numpy arrays.

    Args:
        float_tensor: First tensor to compare.
        fxp_tensor: Second tensor to compare.
        norm: whether to normalize the error function result.
        norm_eps: epsilon value for error normalization stability.

    Returns:
        The MSE distance between the two tensors.
    """
    error = np.power(float_tensor - fxp_tensor, 2.0).mean()
    if norm:
        error /= (np.power(float_tensor, 2.0).mean() + norm_eps)
    return error


def core_config_builder(mixed_precision, num_calibration_iter, num_samples_for_distance, use_grad_based_weights,
                        configuration_overwrite):
    # TODO: Need to edit the config or is default config is enough?
    quant_config = QuantizationConfig()
    mp_config = None
    if mixed_precision:
        # TODO: set distance_fn and after changing default in library
        mp_config = MixedPrecisionQuantizationConfigV2(compute_distance_fn=compute_mse,
                                                       num_of_images=num_samples_for_distance,
                                                       use_grad_based_weights=use_grad_based_weights,
                                                       output_grad_factor=0.0,
                                                       norm_weights=False,
                                                       configuration_overwrite=configuration_overwrite)
    core_config = CoreConfig(num_calibration_iter,
                             quantization_config=quant_config,
                             mixed_precision_config=mp_config)
    return core_config
