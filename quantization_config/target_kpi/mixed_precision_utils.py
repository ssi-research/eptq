import numpy as np
import model_compression_toolkit as mct
from constants import BYTES
from model_compression_toolkit import KPI
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from enum import Enum


class MPCONFIG(Enum):
    MP_PARTIAL_CANDIDATES = 0
    MP_FULL_CANDIDATES = 0


MP_BITWIDTH_OPTIONS_DICT = {
    MPCONFIG.MP_PARTIAL_CANDIDATES: [2, 4, 8],
    MPCONFIG.MP_FULL_CANDIDATES: [2, 3, 4, 5, 6, 7, 8]
}


def get_target_kpi(weights_cr, activation_cr, total_cr, model, representative_data_gen, core_config, tpc):
    kpi_data = mct.keras_kpi_data_experimental(model, representative_data_gen, core_config, DEFAULT_KERAS_INFO, tpc)

    target_weights_kpi = np.inf if weights_cr is None else BYTES * kpi_data.weights_memory / weights_cr
    target_activation_kpi = np.inf if activation_cr is None else BYTES * kpi_data.activation_memory / activation_cr
    target_total_kpi = np.inf if total_cr is None else BYTES * kpi_data.total_memory / total_cr

    return KPI(weights_memory=target_weights_kpi,
               activation_memory=target_activation_kpi,
               total_memory=target_total_kpi)
