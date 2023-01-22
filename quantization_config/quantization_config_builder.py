import model_compression_toolkit
from model_compression_toolkit import CoreConfig, QuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
import numpy as np
import model_compression_toolkit as mct
from tensorflow.keras.layers import Input, Dense
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_last_layer_weights


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


def core_config_builder(mixed_precision, num_samples_for_distance, configuration_overwrite):
    quant_config = QuantizationConfig()
    mp_config = None
    debug_config = model_compression_toolkit.DebugConfig(
        network_editor=[mct.network_editor.EditRule(filter=mct.network_editor.NodeTypeFilter(Input),
                                                    action=mct.network_editor.ChangeCandidatesActivationQuantConfigAttr(
                                                        enable_activation_quantization=False)),
                        mct.network_editor.EditRule(filter=mct.network_editor.NodeTypeFilter(Dense),
                                                    action=mct.network_editor.ChangeCandidatesActivationQuantConfigAttr(
                                                        enable_activation_quantization=False))])
    if mixed_precision:
        mp_config = MixedPrecisionQuantizationConfigV2(compute_distance_fn=None,
                                                       num_of_images=num_samples_for_distance,
                                                       use_grad_based_weights=False,
                                                       distance_weighting_method=get_last_layer_weights,
                                                       configuration_overwrite=configuration_overwrite)
    core_config = CoreConfig(quant_config,
                             mixed_precision_config=mp_config,
                             debug_config=debug_config)
    return core_config
