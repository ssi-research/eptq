import argparse
from constants import VAL_DIR, TRAIN_DIR, ONE_CLASS, DATASET_TYPES, DEFAULT_MP_WEIGHTS_OPTIONS, \
    DEFAULT_MP_ACTIVATION_OPTIONS
from mixed_precision_utils import get_target_kpi
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG
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
    # Run the Model Compression Toolkit
    #################################################
    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.

    if args.mixed_precision is True:
        # TODO: currently running the old, non-experimental MP facade and kpi_data.
        #  Change that after refactors in lib are done.
        mixed_precision_options = [(w, a) for w in DEFAULT_MP_WEIGHTS_OPTIONS for a in DEFAULT_MP_ACTIVATION_OPTIONS]
        # TODO: allow to get candidates from input (also enable quantization)?
        target_platform_model = get_mixed_precision_tp_model(mixed_precision_options=mixed_precision_options)
        target_platform_cap = generate_keras_tpc(target_platform_model, name='mixed_precision_tpc')

        # TODO: Change to core config when lob refactor is completed
        mp_config = DEFAULT_MIXEDPRECISION_CONFIG
        mp_config.num_of_images = args.num_samples_for_distance

        target_kpi = get_target_kpi(args, model, representative_data_gen, target_platform_cap)
        print(f'Target KPI: {target_kpi}')

        quantized_model, quantization_info = \
            mct.keras_post_training_quantization_mixed_precision(model,
                                                                 representative_data_gen,
                                                                 target_kpi=target_kpi,
                                                                 quant_config=mp_config,
                                                                 target_platform_capabilities=target_platform_cap,
                                                                 n_iter=args.num_calibration_iter)
    else:
        target_platform_model = get_fixed_bitwidth_tp_model()
        # TODO: maybe create a dictionary of TP models
        # TODO: allow to get config from input (bitwidth and enable quantization)?
        target_platform_cap = generate_keras_tpc(target_platform_model, name='fixed_bitwidth_tpc')

        quantized_model, quantization_info = mct.keras_post_training_quantization(model,
                                                                                  representative_data_gen,
                                                                                  target_platform_capabilities=target_platform_cap,
                                                                                  n_iter=args.num_calibration_iter)
    #################################################
    # Run accuracy evaluation for the quantized model
    #################################################
    quant_result = model_cfg.evaluation_function(quantized_model, val_dataset)

    print(f'Accuracy of quantized model: {quant_result * 100} (float model: {float_result * 100})')


if __name__ == '__main__':
    main()