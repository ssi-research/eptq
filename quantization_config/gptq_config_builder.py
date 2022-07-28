from typing import List

import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
from utils.radam_optimizer import RAdam
from quantization_config.gptq_loss import GPTQMultipleTensorsLoss
import numpy as np
import wandb


def log_func(loss_value, grads, vars, compare_points):
    results_dict = {v.name + '_grad': np.sqrt(np.sum(np.power(g.numpy(), 2.0)))
                    for g, v in zip(grads, vars) if g is not None}
    results_dict.update({v.name + '_stats': {'RMS': np.sqrt(np.mean(np.power(v.numpy(), 2.0))),
                                             'max': np.abs(v.numpy()).max()}
                         for v in vars if '_auxvar' in v.name})
    results_dict.update({'loss': loss_value.numpy()})
    wandb.log(results_dict)


def build_gptq_config(args):
    rounding_type = RoundingType.STE if args.ste_rounding else RoundingType.GumbelRounding
    optimizer = RAdam(learning_rate=args.lr)
    optimizer_rest = RAdam(learning_rate=1e-4)
    return mct.GradientPTQConfig(n_iter=args.gptq_num_calibration_iter,
                                 optimizer=optimizer,
                                 optimizer_rest=optimizer_rest,
                                 loss=GPTQMultipleTensorsLoss(),
                                 temperature_learning=args.temperature_learning,
                                 train_bias=args.bias_learning,
                                 quantization_parameters_learning=args.quantization_parameters_learning_weights,
                                 rounding_type=rounding_type,
                                 sam_optimization=args.sam_optimization,
                                 rho=args.rho,
                                 log_function=log_func,
                                 use_jac_based_weights=args.jacobian_weights,
                                 num_samples_for_loss=16
                                 )
