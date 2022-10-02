from typing import List
import tensorflow as tf
import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
from utils.radam_optimizer import RAdam
from quantization_config.gptq_loss import GPTQMultipleTensorsLoss
import numpy as np
import wandb


def log_func(loss_value, grads, vars, compare_points):
    results_dict = {}
    # tau_dict = {n: tau.numpy() for n, tau in model_info_dict["tau"].items()}
    #
    # # gt_dict = {gt.name: gt.numpy() for gt in gumbel_temp}
    # gt_res = {k + "_max": np.max(v) for k, v in tau_dict.items()}
    # gt_res.update({k + "_min": np.min(v) for k, v in tau_dict.items()})
    # gt_res.update({k + "_mean": np.mean(v) for k, v in tau_dict.items()})
    # gt_res.update({k + "_var": np.var(v) for k, v in tau_dict.items()})
    results_dict.update({'loss': loss_value.numpy()})
    # results_dict.update(gt_res)
    wandb.log(results_dict)


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
    if args.lr_bias:
        # optimizer_bias = RAdam(learning_rate=args.lr_bias)
        optimizer_bias = tf.keras.optimizers.SGD(learning_rate=args.lr_bias, momentum=0.9)
    else:
        optimizer_bias = None
    if args.lr_quantization_param:
        optimizer_quantization_param = tf.keras.optimizers.SGD(learning_rate=args.lr_quantization_param, momentum=0.9)
    else:
        optimizer_quantization_param = None
    gc = mct.GumbelConfig(temperature_learning=args.temperature_learning, maximal_temp=args.maximal_temp,
                          minimal_temp=args.minimal_temp,
                          gumbel_entropy_regularization=args.gamma_temperature)
    return mct.GradientPTQConfig(n_iter=args.gptq_num_calibration_iter,
                                 optimizer=optimizer,
                                 optimizer_rest=optimizer_rest,
                                 loss=GPTQMultipleTensorsLoss(norm_loss=args.norm_loss),
                                 train_bias=args.bias_learning,
                                 quantization_parameters_learning=args.quantization_parameters_learning,
                                 rounding_type=rounding_type,
                                 sam_optimization=args.sam_optimization,
                                 rho=args.rho,
                                 log_function=log_func,
                                 lsb_change_per_bit_width=build_shift_dict(args),
                                 use_jac_based_weights=args.jacobian_weights,
                                 num_samples_for_loss=args.jacobian_weights_num_samples,
                                 norm_weights=args.norm_weights,
                                 optimizer_bias=optimizer_bias,
                                 optimizer_quantization_parameter=optimizer_quantization_param,
                                 quantizer_config=gc,
                                 gumbel_scale=args.gumbel_scale
                                 )
