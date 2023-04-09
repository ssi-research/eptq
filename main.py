import math
import sys
import argparse
import wandb
import utils
from constants import VAL_DIR, TRAIN_DIR

from models.model_dictionary import model_dictionary
import model_compression_toolkit as mct
import quantization_config
from datetime import datetime

PROJECT_NAME = 'eptq'
FILE_TIME_STAMP = datetime.now().strftime("%d-%b-%Y__%H:%M:%S")


# TODO:
#  1) Remove/update analysis code and save result
#  2) Fix dataset path?
#  3) Update methods comment and typehints
#  4) What to do with timm models copyright message (copied comment in files)
#  6) Add Readme with instructions how to run the basic experiments
#  7) Update requirements


def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the model to run')
    parser.add_argument('--project_name', type=str, default=PROJECT_NAME)
    parser.add_argument('--float_evaluation', action='store_true')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--group', type=str)
    parser.add_argument('--wandb', default=False, action='store_true')

    #####################################################################
    # Dataset Config
    #####################################################################
    parser.add_argument('--val_dataset_folder', type=str, default=VAL_DIR)
    parser.add_argument('--representative_dataset_folder', type=str, default=TRAIN_DIR)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--n_images', type=int, default=1024)

    #####################################################################
    # MCT Config
    #####################################################################
    parser.add_argument('--weights_nbits', type=int, default=4,
                        help='The number of bits for weights quantization')
    parser.add_argument('--activation_nbits', type=int, default=8,
                        help='The number of bits for activation quantization')
    parser.add_argument('--disable_weights_quantization', action='store_true', default=False,
                        help='Flag that disables weights quantization')
    parser.add_argument('--disable_activation_quantization', action='store_true', default=False,
                        help='Flag that disables activation quantization')

    #####################################################################
    # Mixed Precision Config
    #####################################################################
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Enable Mixed-Precision quantization')
    parser.add_argument("--mixed_precision_configuration", nargs="+", default=None,
                        help='Mixed-precision configuration to set to the model instead of searching')
    parser.add_argument('--mp_all_bits', action='store_true', default=False,
                        help='Enable Mixed-Precision quantization')
    parser.add_argument('--weights_cr', type=float,
                        help='Weights compression rate for mixed-precision')
    parser.add_argument('--activation_cr', type=float,
                        help='Activation compression rate for mixed-precision')
    parser.add_argument('--total_cr', type=float,
                        help='Total compression rate for mixed-precision')
    parser.add_argument('--num_samples_for_distance', type=int, default=32,
                        help='Number of samples in distance matrix for distance computation')

    #####################################################################
    # Gumbel Rounding Config
    #####################################################################
    parser.add_argument('--eptq', action='store_true', default=False, help='Enable EPTQ quantization')
    parser.add_argument('--eptq_num_calibration_iter', type=int, default=20000)
    parser.add_argument('--bias_learning', action='store_false', default=True,
                        help='Whether to enable bias learning.')
    parser.add_argument('--quantization_parameters_learning', action='store_false', default=True,
                        help='Whether to enable learning of the quantization threshold.')
    parser.add_argument('--lr', type=float, default=3e-2, help='EPTQ learning rate')
    parser.add_argument('--lr_bias', type=float, default=1e-3, help='Bias learning rate')
    parser.add_argument('--lr_quantization_param', type=float, default=1e-3,
                        help='Threshold learning rate')
    parser.add_argument('--lr_rest', type=float, default=1e-3, help='Learning rate for additional learnable parameters')
    parser.add_argument('--reg_factor', type=float, default=0.01,
                        help='regularization hyper-parameter for GPTQ soft quantizer')

    # Loss
    parser.add_argument('--norm_loss', action='store_true', default=False,
                        help='Whether to normalize the loss value in GPTQ training.')
    parser.add_argument('--hessian_weights', action='store_true', default=False,
                        help='Whether to use the Hessian-based weights in the optimization loss function computation.')
    parser.add_argument('--hessian_weights_num_samples', type=int, default=16,
                        help='Number of samples to be used for Hessian-based weights computation.')
    parser.add_argument('--hessian_weights_num_iter', type=int, default=100,
                        help='Number of iterations to run the Hessian approximation.')
    parser.add_argument('--norm_weights', action='store_true', default=False,
                        help='Whether to normalize the Hessian-based loss weights.')
    parser.add_argument('--scale_log_norm', action='store_true', default=False)

    args = parser.parse_args()
    return args


def get_float_result(in_args, in_model_cfg, in_model, in_val_dataset) -> float:
    #################################################
    # Run accuracy evaluation for the float model
    #################################################
    if in_args.float_evaluation:
        float_result = in_model_cfg.evaluation_function(in_model, in_val_dataset)
        print(
            f'Float evaluation result: {float_result * 100} (saved float result {in_model_cfg.get_float_accuracy() * 100})')
    else:
        float_result = in_model_cfg.get_float_accuracy()
        print(f'Saved float result: {float_result}')
    return float_result


def main():
    args = argument_handler()
    group = None
    name = None
    if args.group is not None:
        group = f"{args.model_name}_{args.eptq}_{args.mixed_precision}_{args.group}"
        name = f"{args.model_name}_{FILE_TIME_STAMP}"
    if args.wandb:
        wandb.init(project=PROJECT_NAME, group=group, name=name)
        wandb.config.update(args)
    utils.set_seed(args.random_seed)

    #################################################
    # Build quantization configuration
    #################################################
    configuration_override = None
    if args.mixed_precision_configuration is not None:
        configuration_override = [int(b) for b in args.mixed_precision_configuration]

    core_config = quantization_config.core_config_builder(args.mixed_precision,
                                                          args.num_samples_for_distance,
                                                          configuration_override)

    #################################################
    # Run the Model Compression Toolkit
    #################################################
    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.
    mixed_precision_config = utils.MPCONFIG.MP_FULL_CANDIDATES if args.mp_all_bits else utils.MPCONFIG.MP_PARTIAL_CANDIDATES

    target_platform_cap, bit_width_mapping = quantization_config.build_target_platform_capabilities(
        args.mixed_precision,
        args.activation_nbits,
        args.weights_nbits,
        args.disable_weights_quantization,
        args.disable_activation_quantization,
        args.weights_cr, args.activation_cr,
        args.total_cr,
        mixed_precision_config=mixed_precision_config)

    #################################################
    # Generate Model
    #################################################
    model_cfg = model_dictionary.get(args.model_name)
    model = model_cfg.get_model()

    #################################################
    # Floating-point accuracy
    #################################################
    val_dataset = model_cfg.get_validation_dataset(
        dir_path=args.val_dataset_folder,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size))
    float_result = get_float_result(args, model_cfg, model, val_dataset)

    #################################################
    # Get datasets
    #################################################
    n_iter = math.ceil(args.n_images // args.batch_size)
    representative_data_gen = model_cfg.get_representative_dataset(
        representative_dataset_folder=args.representative_dataset_folder,
        n_iter=n_iter,
        batch_size=args.batch_size,
        n_images=args.n_images,
        image_size=args.image_size,
        preprocessing=None,
        seed=args.random_seed)

    target_kpi, full_kpi = quantization_config.build_target_kpi(args.weights_cr, args.activation_cr, args.total_cr,
                                                                args.mixed_precision, model, representative_data_gen,
                                                                core_config,
                                                                target_platform_cap)

    if args.eptq:
        gptq_config = quantization_config.build_gptq_config(args, n_iter)

        quantized_model, quantization_info = \
            mct.gptq.keras_gradient_post_training_quantization_experimental(model,
                                                                            representative_data_gen,
                                                                            gptq_config=gptq_config,
                                                                            target_kpi=target_kpi,
                                                                            core_config=core_config,
                                                                            target_platform_capabilities=target_platform_cap)
    else:
        quantized_model, quantization_info = \
            mct.keras_post_training_quantization_experimental(model,
                                                              representative_data_gen,
                                                              target_kpi=target_kpi,
                                                              core_config=core_config,
                                                              target_platform_capabilities=target_platform_cap)

    #################################################
    # Run accuracy evaluation for the quantized model
    #################################################
    quant_result = model_cfg.evaluation_function(quantized_model, val_dataset)

    if args.wandb:
        wandb.config.update({"mixed_precision_cfg_final": quantization_info.mixed_precision_cfg,
                             "bit-width-mapping": bit_width_mapping})
        wandb.log({"quantized_results": quant_result * 100,
                     "float_results": float_result * 100,
                     **quantization_config.kpi2dict(target_kpi),
                     **quantization_config.kpi2dict(quantization_info.final_kpi, "final"),
                     **quantization_config.kpi2dict(full_kpi, "max_kp")})

    print(f'Accuracy of quantized model: {quant_result * 100} (float model: {float_result * 100})')


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
