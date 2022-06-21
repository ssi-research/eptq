import argparse
from constants import VAL_DIR, TRAIN_DIR, ONE_CLASS, DATASET_TYPES
from datasets.gen_representative_datasets import get_representative_dataset
from model_configs.model_dictionary import model_dictionary
import model_compression_toolkit as mct


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
    representative_data_gen = get_representative_dataset(args)

    #################################################
    # Run the Model Compression Toolkit
    #################################################
    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.
    # Here, for example, we use the default target platform model that is attached to a Tensorflow
    # layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default')

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