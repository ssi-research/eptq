import argparse
from constants import VAL_DIR, TRAIN_DIR, ONE_CLASS, DATASET_TYPES, MP_PARTIAL_CANDIDATES
from mixed_precision_utils import get_target_kpi, MP_BITWIDTH_OPTIONS_DICT
from model_compression_toolkit import CoreConfig, QuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_configs.model_dictionary import model_dictionary
import model_compression_toolkit as mct
from target_platform.fix_bitwidth_tp_model import get_fixed_bitwidth_tp_model
from target_platform.mixed_precision_tp_model import get_mixed_precision_tp_model
from target_platform.tensorflow_tpc import generate_keras_tpc


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the model to run')
    parser.add_argument('--project_name', type=str, default='gumbel-rounding')
    parser.add_argument('--val_dataset_folder', type=str, default=VAL_DIR)
    parser.add_argument('--representative_dataset_folder', type=str, default=TRAIN_DIR)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_calibration_iter', type=int, default=1)
    parser.add_argument('--float_evaluation', action='store_true')
    parser.add_argument('--dataset_type', type=str, default=ONE_CLASS,
                        choices=DATASET_TYPES)
    parser.add_argument('--n_images', type=int, default=1)
    parser.add_argument('--img_num', type=int, default=0)
    parser.add_argument('--class_num', type=int, default=0)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--random_batch', action='store_true')

    parser.add_argument('--weights_nbits', type=int, default=8,
                        help='The number of bits for weights quantization')
    parser.add_argument('--activation_nbits', type=int, default=8,
                        help='The number of bits for activation quantization')
    parser.add_argument('--disable_weights_quantization', action='store_true', default=False,
                        help='Flag that disables weights quantization')
    parser.add_argument('--disable_activation_quantization', action='store_true', default=False,
                        help='Flag that disables activation quantization')

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

    args = parser.parse_args()
    return args


def main():
    args = argument_handler()
    model_cfg = model_dictionary.get(args.model_name)
    model = model_cfg.get_model()
    val_dataset = model_cfg.get_dataset(
        dir_path=args.val_dataset_folder,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        shuffle=False)

    #################################################
    # Run accuracy evaluation for the float model
    #################################################
    if args.float_evaluation:
        float_result = model_cfg.evaluation_function(model, val_dataset)
        print(f'Float evaluation result: {float_result * 100} (saved float result {model_cfg.get_float_accuracy() * 100})')
    else:
        float_result = model_cfg.get_float_accuracy()
        print(f'Saved float result: {float_result}')

    #################################################
    # Get datasets
    #################################################
    representative_data_gen = model_cfg.get_representative_dataset(args)

    #################################################
    # Build quantization configuration
    #################################################
    # TODO: Need to edit the config or is default config is enough?
    quant_config = QuantizationConfig()
    mp_config = None
    if args.mixed_precision:
        # TODO: set distance_fn and after changing default in library
        mp_config = MixedPrecisionQuantizationConfigV2(num_of_images=args.num_samples_for_distance,
                                                       use_grad_based_weights=args.use_grad_based_weights)
    core_config = CoreConfig(args.num_calibration_iter,
                             quantization_config=quant_config,
                             mixed_precision_config=mp_config)

    #################################################
    # Run the Model Compression Toolkit
    #################################################
    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.

    target_kpi = None
    if args.mixed_precision is True:
        # TODO: allow to choose different candidates
        mixed_precision_options = [(w, a)
                                   for w in MP_BITWIDTH_OPTIONS_DICT[MP_PARTIAL_CANDIDATES]
                                   for a in MP_BITWIDTH_OPTIONS_DICT[MP_PARTIAL_CANDIDATES]]
        target_platform_model = get_mixed_precision_tp_model(mixed_precision_options=mixed_precision_options,
                                                             enable_weights_quantization=not args.disable_weights_quantization,
                                                             enable_activation_quantization=not args.disable_activation_quantization)
        target_platform_cap = generate_keras_tpc(target_platform_model, name='mixed_precision_tpc')

        target_kpi = get_target_kpi(args, model, representative_data_gen, core_config, target_platform_cap)
        print(f'Target KPI: {target_kpi}')

    else:
        # TODO: maybe create a dictionary of TP models
        # TODO: allow to get config from input (bitwidth and enable quantization)?
        target_platform_model = get_fixed_bitwidth_tp_model(weights_n_bits=args.weights_nbits,
                                                            activation_n_bits=args.activation_nbits,
                                                            enable_weights_quantization=not args.disable_weights_quantization,
                                                            enable_activation_quantization=not args.disable_activation_quantization)
        target_platform_cap = generate_keras_tpc(target_platform_model, name='fixed_bitwidth_tpc')

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