import numpy as np
import model_compression_toolkit as mct
from constants import BYTES, MP_PARTIAL_CANDIDATES, MP_FULL_CANDIDATES
from model_compression_toolkit import KPI
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO


MP_BITWIDTH_OPTIONS_DICT = {
    MP_PARTIAL_CANDIDATES: [2, 4, 8],
    MP_FULL_CANDIDATES: [2, 3, 4, 5, 6, 7, 8]
}


def get_target_kpi(args, model, representative_data_gen, core_config, tpc):
    kpi_data = mct.keras_kpi_data_experimental(model, representative_data_gen, core_config, DEFAULT_KERAS_INFO, tpc)

    target_weights_kpi = np.inf if args.weights_cr is None else BYTES * kpi_data.weights_memory / args.weights_cr
    target_activation_kpi = np.inf if args.activation_cr is None else BYTES * kpi_data.activation_memory / args.activation_cr
    target_total_kpi = np.inf if args.total_cr is None else BYTES * kpi_data.activation_memory / args.total_cr

    return KPI(weights_memory=target_weights_kpi,
               activation_memory=target_activation_kpi,
               total_memory=target_total_kpi)
