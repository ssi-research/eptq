import tensorflow as tf
import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
from utils.radam_optimizer import RAdam
from quantization_config.gptq_loss import GPTQMultipleTensorsLoss
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

    gumbel_scale_per_bitwidth = None
    if args.gumbel_scale_per_bitwidth is not None:
        gumbel_scale_per_bitwidth = {}
        assert len(args.gumbel_scale_per_bitwidth) == 3, "To use different gumbel scale value per bit-width you " \
                                                         "should provide a list with 3 values under argument " \
                                                         "gumbel_scale_per_bitwidth - for 2, 4 and 8 bit (in this order)."
        gumbel_scale_per_bitwidth[2] = float(args.gumbel_scale_per_bitwidth[0])
        gumbel_scale_per_bitwidth[4] = float(args.gumbel_scale_per_bitwidth[1])
        gumbel_scale_per_bitwidth[8] = float(args.gumbel_scale_per_bitwidth[2])

    gc = mct.GumbelConfig(temperature_learning=args.temperature_learning,
                          maximal_temp=args.maximal_temp,
                          minimal_temp=args.minimal_temp,
                          gumbel_entropy_regularization=args.gamma_temperature,
                          gumbel_scale=args.gumbel_scale,
                          gumbel_scale_per_bitwidth=gumbel_scale_per_bitwidth)

    return mct.GradientPTQConfigV2(n_epochs=args.gptq_num_calibration_iter,
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
                                   log_norm=args.gptq_log_norm,
                                   weights_n_iter=args.jacobian_weights_num_iter,
                                   )
