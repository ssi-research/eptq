import argparse
import wandb
import utils
from constants import VAL_DIR, TRAIN_DIR

from models.model_dictionary import model_dictionary
import model_compression_toolkit as mct
import quantization_config
from datetime import datetime
from augmentation.augmenetation_piple import generate_augmentation_function

PROJECT_NAME = 'gumbel-rounding'
FILE_TIME_STAMP = datetime.now().strftime("%d-%b-%Y__%H:%M:%S")
#
MPOVERRIDE_DICT_W = {"resnet18": {8: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
                                  8.8: [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                                  9.8: [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1],
                                  11: [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1],
                                  12.5: [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1]},
                     "mbv2": {

                         8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                             1, 0, 0, 2, 1, 1, 2, 0, 1, 1, 1, 0, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],

                         8.8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                               0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], },

                     "regnetx_006": {
                         8: [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
                         8.8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1],
                         9.8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1],
                         11: [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1],
                         12.5: [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1,
                                2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

                     }

                     }
# MPOVERRIDE_DICT_T = {"resnet18": {
#     8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 1, 3, 0, 3, 0, 1, 0, 3, 0, 1, 3, 0, 3, 0, 1, 0, 6,
#         0, 3]}}


MPOVERRIDE_DICT_T = {"regnetx_006": {
    6: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 1,
        0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 0, 0, 1, 0, 3, 0, 0],
    7: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 3, 0, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1,
        0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 3],
    8: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 3, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 6, 0, 1,
        0, 1, 0, 6, 0, 1, 0, 1, 0, 6, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 2, 0, 3, 0, 3],
    8.8: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 3, 0, 0, 0, 0, 1, 0, 3, 0, 1,
          0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 3, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 6, 0, 1,
          0, 1, 0, 6, 0, 1, 0, 1, 0, 6, 0, 1, 0, 1, 0, 6, 0, 1, 0, 1, 0, 6, 0, 1, 0, 2, 0, 3, 0, 3],
    9.8: [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 3, 0, 1,
          0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 3, 0, 3, 1, 0, 1, 0, 3, 0, 2, 0, 1, 0, 6, 0, 2,
          0, 1, 0, 6, 0, 2, 0, 1, 0, 6, 0, 1, 0, 1, 0, 6, 0, 1, 0, 1, 0, 6, 0, 1, 0, 2, 0, 3, 0, 3],
    11: [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1],
    12.5: [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1,
           2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

}}


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
    parser.add_argument('--debug', action='store_true')

    #####################################################################
    # Dataset Config
    #####################################################################
    parser.add_argument('--val_dataset_folder', type=str, default=VAL_DIR)
    parser.add_argument('--representative_dataset_folder', type=str, default=TRAIN_DIR)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--n_images', type=int, default=1024)

    #####################################################################
    # Augmentations
    #####################################################################
    parser.add_argument('--disable_augmentations', action='store_false', default=True)
    parser.add_argument('--aug_mean', nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--aug_std', nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--aug_alpha', type=float, default=0.25)
    parser.add_argument('--aug_p', type=float, default=None)
    parser.add_argument('--aug_dequantization', action='store_true', default=False)

    #####################################################################
    # MCT Config
    #####################################################################
    parser.add_argument('--num_calibration_iter', type=int, default=10)
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
    parser.add_argument('--mixed_precision_override', action='store_true', default=False,
                        help='Enable Mixed-Precision quantization')
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
    parser.add_argument('--use_grad_based_weights', action='store_true', default=False,
                        help='A flag to enable gradient-based weights for distance metric weighted average')
    parser.add_argument('--dense2bit', action='store_true', default=False,
                        help='Enable Mixed-Precision quantization')
    #####################################################################
    # Gumbel Rounding Config
    #####################################################################
    parser.add_argument('--gptq', action='store_true', default=False, help='Enable GPTQ quantization')
    parser.add_argument('--gptq_num_calibration_iter', type=int, default=40000)
    parser.add_argument('--ste_rounding', action='store_true', default=False)
    parser.add_argument('--sam_optimization', action='store_true', default=False)
    parser.add_argument('--temperature_learning', action='store_true', default=False)
    parser.add_argument('--bias_learning', action='store_true', default=False)
    parser.add_argument('--is_symmetric', action='store_true', default=False)
    parser.add_argument('--is_symmetric_activation', action='store_true', default=False)
    parser.add_argument('--quantization_parameters_learning', action='store_true', default=False)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--minimal_temp', type=float, default=0.1)
    parser.add_argument('--maximal_temp', type=float, default=0.5)
    parser.add_argument('--gamma_temperature', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.15, help='GPTQ learning rate')
    parser.add_argument('--lr_rest', type=float, default=1e-4, help='GPTQ learning rate')
    parser.add_argument('--lr_bias', type=float, default=1e-4, help='GPTQ learning rate')
    parser.add_argument('--lr_quantization_param', type=float, default=1e-3, help='GPTQ learning rate')
    parser.add_argument('--gumbel_scale', type=float, default=1.0, help='Gumbel randomization tensor factor')
    parser.add_argument('--disable_activation_quantization_gptq', action='store_true', default=False,
                        help='Enable GPTQ quantization')

    parser.add_argument('--m8', type=int, default=1)
    parser.add_argument('--m7', type=int, default=1)
    parser.add_argument('--m6', type=int, default=1)
    parser.add_argument('--m5', type=int, default=0)
    parser.add_argument('--m4', type=int, default=0)
    parser.add_argument('--m3', type=int, default=0)
    parser.add_argument('--m2', type=int, default=0)

    # Loss
    parser.add_argument('--hessian_weighting', action='store_true', default=False)
    parser.add_argument('--bn_p_norm', action='store_true', default=False)
    parser.add_argument('--activation_bias', action='store_true', default=False)
    parser.add_argument('--norm_loss', action='store_true', default=False)
    parser.add_argument('--jacobian_weights', action='store_true', default=False)
    parser.add_argument('--jacobian_weights_num_samples', type=int, default=16)
    parser.add_argument('--norm_weights', action='store_true', default=False)

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
        group = f"{args.model_name}_{args.gptq}_{args.mixed_precision}_{args.group}"
        name = f"{args.model_name}_{FILE_TIME_STAMP}"
    if not args.debug:
        wandb.init(project=PROJECT_NAME, group=group, name=name)
        wandb.config.update(args)
    utils.set_seed(args.random_seed)

    #################################################
    # Build quantization configuration
    #################################################
    configuration_overwrite = None
    if args.mixed_precision_override:
        weights_mp = args.weights_cr is not None or args.total_cr is not None
        activation_mp = args.activation_cr is not None or args.total_cr is not None
        if weights_mp and activation_mp:
            configuration_overwrite = MPOVERRIDE_DICT_T[args.model_name][args.total_cr]
        elif weights_mp:
            configuration_overwrite = MPOVERRIDE_DICT_W[args.model_name][args.weights_cr]
        else:
            raise NotImplemented
    core_config = quantization_config.core_config_builder(args.mixed_precision,
                                                          args.num_samples_for_distance,
                                                          args.use_grad_based_weights,
                                                          configuration_overwrite)

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
        mixed_precision_config=mixed_precision_config,
        is_symmetric=args.is_symmetric, is_symmetric_act=args.is_symmetric_activation)
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
    if args.disable_augmentations:
        augmentation_pipeline = None
    else:
        augmentation_pipeline = generate_augmentation_function(tuple(args.aug_mean), tuple(args.aug_std),
                                                               args.aug_alpha, args.aug_p, args.aug_dequantization)

    representative_data_gen = model_cfg.get_representative_dataset(
        representative_dataset_folder=args.representative_dataset_folder,
        n_iter=args.num_calibration_iter,
        batch_size=args.batch_size,
        n_images=args.n_images,
        image_size=args.image_size,
        preprocessing=None,
        seed=args.random_seed,
        debug=args.debug,
        augmentation_pipepline=augmentation_pipeline)

    target_kpi, full_kpi = quantization_config.build_target_kpi(args.weights_cr, args.activation_cr, args.total_cr,
                                                                args.mixed_precision, model, representative_data_gen,
                                                                core_config,
                                                                target_platform_cap)

    if args.gptq:
        gptq_config = quantization_config.build_gptq_config(args)

        quantized_model, quantization_info = \
            mct.keras_gradient_post_training_quantization_experimental(model,
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
    wandb.config.update({"mixed_precision_cfg_final": quantization_info.mixed_precision_cfg,
                         "bit-width-mapping": bit_width_mapping})
    wandb.log({"quantized_results": quant_result * 100,
               "float_results": float_result * 100,
               **quantization_config.kpi2dict(target_kpi),
               **quantization_config.kpi2dict(quantization_info.final_kpi, "final"),
               **quantization_config.kpi2dict(full_kpi, "max_kp")})
    print(f'Accuracy of quantized model: {quant_result * 100} (float model: {float_result * 100})')


if __name__ == '__main__':
    main()
