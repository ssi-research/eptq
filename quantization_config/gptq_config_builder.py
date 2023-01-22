import tensorflow as tf
import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
from utils.radam_optimizer import RAdam
from quantization_config.gptq_loss import GPTQMultipleTensorsLoss
import numpy as np
import wandb


def log_func(loss_value):
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

    quantizer_config = mct.SoftQuantizerConfig(num_batches=n_iter, entropy_regularization=args.gamma_temperature)

    return mct.GradientPTQConfigV2(n_epochs=int(np.ceil(args.gptq_num_calibration_iter/n_iter)),
                                   optimizer=optimizer,
                                   optimizer_rest=optimizer_rest,
                                   loss=GPTQMultipleTensorsLoss(norm_loss=args.norm_loss),
                                   train_bias=args.bias_learning,
                                   quantization_parameters_learning=args.quantization_parameters_learning,
                                   rounding_type=RoundingType.SoftQuantizer,
                                   log_function=log_func,
                                   use_jac_based_weights=args.hessian_weights,
                                   num_samples_for_loss=args.hessian_weights_num_samples,
                                   norm_weights=args.norm_weights,
                                   optimizer_bias=optimizer_bias,
                                   optimizer_quantization_parameter=optimizer_quantization_param,
                                   quantizer_config=quantizer_config,
                                   log_norm=True,
                                   weights_n_iter=args.hessian_weights_num_iter)
