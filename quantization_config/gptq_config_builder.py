from typing import List

import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
from utils.radam_optimizer import RAdam
from quantization_config.gptq_loss import GPTQMultipleTensorsLoss


def build_gptq_config(args):
    rounding_type = RoundingType.STE if args.ste_rounding else RoundingType.GumbelRounding

    return mct.GradientPTQConfig(n_iter=args.gptq_num_calibration_iter,
                                 optimizer=RAdam(learning_rate=args.lr,
                                                 total_steps=args.gptq_num_calibration_iter,
                                                 warmup_proportion=0.2),
                                 loss=GPTQMultipleTensorsLoss(),
                                 log_function=None,  # TODO: add logging function when adding WANDB support
                                 train_bias=args.bias_learning,
                                 quantization_parameters_learning=args.quantization_parameters_learning_weights,
                                 temperature_learning=args.temperature_learning,
                                 sam_optimization=args.sam_optimization,
                                 rounding_type=rounding_type,
                                 rho=args.rho,
                                 gumbel_entropy_regularization=args.gamma_temperature)
