from quantization_config.target_kpi.mixed_precision_utils import get_target_kpi


def build_target_kpi(weights_cr, activation_cr, total_cr, mixed_precision, model, representative_data_gen, core_config,
                     target_platform_cap):
    target_kpi = None
    if mixed_precision is True:
        target_kpi = get_target_kpi(weights_cr, activation_cr, total_cr, model, representative_data_gen, core_config,
                                    target_platform_cap)
        print(f'Target KPI: {target_kpi}')
    return target_kpi
