import argparse
from constants import VAL_DIR, TRAIN_DIR

from model_configs.model_dictionary import model_dictionary
import model_compression_toolkit as mct
import quantization_config

PROJECT_NAME = 'gumbel-rounding'


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
    #####################################################################
    # Dataset Config
    #####################################################################
    parser.add_argument('--val_dataset_folder', type=str, default=VAL_DIR)
    parser.add_argument('--representative_dataset_folder', type=str, default=TRAIN_DIR)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--n_images', type=int, default=1024)
    #####################################################################
    # MCT Config
    #####################################################################
    parser.add_argument('--num_calibration_iter', type=int, default=100)
    parser.add_argument('--weights_nbits', type=int, default=8,
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
    #####################################################################
    # Gumbel Rounding Config
    #####################################################################
    parser.add_argument('--gptq', action='store_true', default=False,
                        help='Enable GPTQ quantization')
    parser.add_argument('--gptq_num_calibration_iter', type=int, default=20000)
    parser.add_argument('--ste_rounding', action='store_true', default=False)
    parser.add_argument('--sam_optimization', action='store_true', default=False)
    parser.add_argument('--temperature_learning', action='store_true', default=False)
    parser.add_argument('--bias_learning', action='store_true', default=False)
    parser.add_argument('--quantization_parameters_learning_weights', action='store_true', default=False)
    parser.add_argument('--quantization_parameters_learning_activation', action='store_true', default=False)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--gamma_temperature', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='GPTQ learning rate')
    # Loss
    parser.add_argument('--hessian_weighting', action='store_true', default=False)
    parser.add_argument('--bn_p_norm', action='store_true', default=False)
    parser.add_argument('--activation_bias', action='store_true', default=False)

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
    model_cfg = model_dictionary.get(args.model_name)
    model = model_cfg.get_model()
    val_dataset = model_cfg.get_validation_dataset(
        dir_path=args.val_dataset_folder,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size))
    float_result = get_float_result(args, model_cfg, model, val_dataset)
    #################################################
    # Get datasets
    #################################################
    representative_data_gen = model_cfg.get_representative_dataset(
        representative_dataset_folder=args.representative_dataset_folder,
        batch_size=args.batch_size,
        n_images=args.n_images,
        image_size=args.image_size,
        preprocessing=None, seed=args.random_seed)

    #################################################
    # Build quantization configuration
    #################################################
    core_config = quantization_config.core_config_builder(args.mixed_precision, args.num_calibration_iter,
                                                          args.num_samples_for_distance, args.use_grad_based_weights)

    #################################################
    # Run the Model Compression Toolkit
    #################################################
    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.
    target_platform_cap = quantization_config.build_target_platform_capabilities(args.mixed_precision, args.activation_nbits,
                                                                                 args.weights_nbits,
                                                                                 args.disable_weights_quantization,
                                                                                 args.disable_activation_quantization)

    target_kpi = quantization_config.build_target_kpi(args.weights_cr, args.activation_cr, args.total_cr,
                                                      args.mixed_precision, model, representative_data_gen, core_config,
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

    print(f'Accuracy of quantized model: {quant_result * 100} (float model: {float_result * 100})')


if __name__ == '__main__':
    main()
