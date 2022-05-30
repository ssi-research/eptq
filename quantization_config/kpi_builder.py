import numpy as np
import model_compression_toolkit as mct
from constants import BYTES
from model_compression_toolkit import KPI
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO


def build_target_kpi(weights_cr, activation_cr, total_cr, mixed_precision, model, representative_data_gen, core_config,
                     target_platform_cap):
    target_kpi = None
    fully_kpi = None
    if mixed_precision is True:
        target_kpi, fully_kpi = get_target_kpi(weights_cr, activation_cr, total_cr, model, representative_data_gen,
                                               core_config,
                                               target_platform_cap)
        print(f'Target KPI: {target_kpi}')
    return target_kpi, fully_kpi


def get_target_kpi(weights_cr, activation_cr, total_cr, model, representative_data_gen, core_config, tpc):
    kpi_data = mct.keras_kpi_data_experimental(model, representative_data_gen, core_config, DEFAULT_KERAS_INFO, tpc)

    target_weights_kpi = np.inf if weights_cr is None else BYTES * kpi_data.weights_memory / weights_cr
    target_activation_kpi = np.inf if activation_cr is None else BYTES * kpi_data.activation_memory / activation_cr
    target_total_kpi = np.inf if total_cr is None else BYTES * kpi_data.total_memory / total_cr

    return KPI(weights_memory=target_weights_kpi,
               activation_memory=target_activation_kpi,
               total_memory=target_total_kpi), kpi_data


def kpi2dict(in_kpi: KPI, add_tag: str = "") -> dict:
    if in_kpi is None:
        return {f"{add_tag}_weights_memory": None,
                f"{add_tag}_activation_memory": None,
                f"{add_tag}_total_memory": None}
    return {f"{add_tag}_weights_memory": in_kpi.weights_memory,
            f"{add_tag}_activation_memory": in_kpi.activation_memory,
            f"{add_tag}_total_memory": in_kpi.total_memory}
