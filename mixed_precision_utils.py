import numpy as np
import model_compression_toolkit as mct
from constants import BYTES
from model_compression_toolkit import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO


def get_target_kpi(args, model, representative_data_gen, tpc):
    # TODO: Change to non-default configs and fw_info when lib refactors are sone and when we decide on the requested config
    kpi_data = mct.keras_kpi_data(model, representative_data_gen, DEFAULT_MIXEDPRECISION_CONFIG, DEFAULT_KERAS_INFO, tpc)

    target_weights_kpi = np.inf if args.weights_cr is None else BYTES * kpi_data.weights_memory / args.weights_cr
    target_activation_kpi = np.inf if args.activation_cr is None else BYTES * kpi_data.activation_memory / args.activation_cr
    target_total_kpi = np.inf if args.total_cr is None else BYTES * kpi_data.activation_memory / args.total_cr

    return KPI(weights_memory=target_weights_kpi,
               activation_memory=target_activation_kpi,
               total_memory=target_total_kpi)
