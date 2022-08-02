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


from model_compression_toolkit.core.common.defaultdict import DefaultDict


def build_shift_dict(args):
    shift_dict = {8: args.m8,
                  7: args.m7,
                  6: args.m6,
                  5: args.m5,
                  4: args.m4,
                  3: args.m3,
                  2: args.m2}
    return shift_dict


def build_gptq_config(args):
    rounding_type = RoundingType.STE if args.ste_rounding else RoundingType.GumbelRounding
    optimizer = RAdam(learning_rate=args.lr)
    optimizer_rest = RAdam(learning_rate=args.lr_rest)
    return mct.GradientPTQConfig(n_iter=args.gptq_num_calibration_iter,
                                 optimizer=optimizer,
                                 optimizer_rest=optimizer_rest,
                                 loss=GPTQMultipleTensorsLoss(norm_loss=args.norm_loss),
                                 temperature_learning=args.temperature_learning,
                                 train_bias=args.bias_learning,
                                 quantization_parameters_learning=args.quantization_parameters_learning_weights,
                                 rounding_type=rounding_type,
                                 sam_optimization=args.sam_optimization,
                                 rho=args.rho,
                                 log_function=log_func,
                                 lsb_change_per_bit_width=build_shift_dict(args),
                                 use_jac_based_weights=args.jacobian_weights,
                                 num_samples_for_loss=args.jacobian_weights_num_samples,
                                 norm_weights=args.norm_weights
                                 )
