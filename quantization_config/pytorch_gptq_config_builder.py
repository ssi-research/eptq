import torch
import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
from torch.optim import RAdam, SGD
from quantization_config.pytorch_gptq_loss import GPTQMultipleTensorsLoss
import wandb


def log_func(loss_value, grads, vars):
    results_dict = {'loss': loss_value}
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


def pytorch_build_gptq_config(args):
    rounding_type = RoundingType.STE if args.ste_rounding else RoundingType.GumbelRounding
    optimizer = RAdam([torch.Tensor([])], lr=args.lr, eps=1e-07)
    optimizer_rest = RAdam([torch.Tensor([])], lr=args.lr_rest, eps=1e-07)
    if args.lr_bias:
        optimizer_bias = SGD([torch.Tensor([])], lr=args.lr_bias, momentum=0.9)
    else:
        optimizer_bias = None
    if args.lr_quantization_param:
        optimizer_quantization_param = SGD([torch.Tensor([])], lr=args.lr_quantization_param, momentum=0.9)
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
