# EPTQ
Keras implementation of EPTQ.
EPTQ is a post-training quantization method for CV networks.
It uses the network's loss function Hessian trace to preform an adaptive optimization for the rounding error of the 
quantized parameters, without the need of a labeled dataset.


## Models

We provide a large set of pretrained models. 
The models are based on their implementation in the 
[tensorflow-image-models](https://github.com/martinsbruveris/tensorflow-image-models) repository and were 
modified for our purposes.

The names of the available models can be found under [models/models_dictionay.py](./models/models_dictionay.py).
It includes the following models:

| Model        | Model usage name |
|--------------|------------------|
| ResNet18     | resnet18         |
| ResNet50     | resnet50         |
| RegNet-600MF | regnetx_006      |
| MobileNet-V2 | mbv2             |
| DeiT-B       | deit             |
| Mixer-B/16   | mlp_mixer        |

## Setup

`pip install -r requirements.txt`

**Note:** our implementation of EPTQ is based on the tools provided by 
[**Model-Compression-Toolkit (MCT)**](https://github.com/sony/model_optimization) version 1.9.

## Usage

### Single precision quantization
`python main.py -m resnet18 --weights_nbits 4 --activation_nbits 8 --train_data_path <path_to_training_dataset> 
--val_data_path <path_to_validation_dataset>`

This example would execute EPTQ via MCT to quantize ResNet18 with 4 bits for weights and 8 bits for activation, 
using 1,024 training samples.

For faster execution you can reduce the number of optimization steps, using the flag 
`--eptq_num_calibration_iter` (80K by default). This might result in small reduction to the final accuracy results.

### Mixed precision quantization
We also enable mixed precision quantization, in which different weights or activation tensors can be quantized 
with different bit-width, under pre-defined memory restrictions 
(weights compression, activation compression or total compression).

`python main.py -m regnetx_006 --mixed_precision --total_cr 8 --train_data_path <path_to_training_dataset> 
--val_data_path <path_to_validation_dataset>`

This example would execute EPTQ via MCT to quantize RegNet-600MF with the bit-width options of 2, 4 and 8 bits for 
both weights and activations, while compressing the **total** memory of the model by a factor of 8.

