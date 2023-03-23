import tensorflow as tf
import model_compression_toolkit as mct
from model_compression_toolkit.gptq import RoundingType
from model_compression_toolkit.gptq.common.gptq_config import GPTQHessianWeightsConfig
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR
from utils.radam_optimizer import RAdam
from quantization_config.gptq_loss import GPTQMultipleTensorsLoss
import numpy as np
import wandb


def log_func(loss_value, grads, vars, compare_points):
    results_dict = {}
    results_dict.update({'loss': loss_value.numpy()})
    wandb.log(results_dict)


def build_gptq_config(args, n_iter):
    optimizer = RAdam(learning_rate=args.lr)
    optimizer_rest = RAdam(learning_rate=args.lr_rest)
    if args.lr_bias:
        optimizer_bias = tf.keras.optimizers.SGD(learning_rate=args.lr_bias, momentum=0.9)
    else:
        optimizer_bias = None
    if args.lr_quantization_param:
        optimizer_quantization_param = tf.keras.optimizers.SGD(learning_rate=args.lr_quantization_param, momentum=0.9)
    else:
        optimizer_quantization_param = None

    hessians_weights_config = GPTQHessianWeightsConfig(hessians_num_samples=args.hessian_weights_num_samples,
                                                       norm_weights=args.norm_weights,
                                                       log_norm=True,
                                                       scale_log_norm=args.scale_log_norm,
                                                       hessians_n_iter=args.hessian_weights_num_iter)

    return mct.gptq.GradientPTQConfigV2(n_epochs=int(np.ceil(args.gptq_num_calibration_iter / n_iter)),
                                        optimizer=optimizer,
                                        optimizer_rest=optimizer_rest,
                                        loss=GPTQMultipleTensorsLoss(norm_loss=args.norm_loss),
                                        train_bias=args.bias_learning,
                                        rounding_type=RoundingType.SoftQuantizer,
                                        log_function=log_func,
                                        optimizer_bias=optimizer_bias,
                                        optimizer_quantization_parameter=optimizer_quantization_param,
                                        regularization_factor=args.reg_factor,
                                        hessian_weights_config=hessians_weights_config,
                                        gptq_quantizer_params_override={QUANT_PARAM_LEARNING_STR:
                                                                            args.quantization_parameters_learning})
