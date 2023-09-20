# EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian
Keras implementation of EPTQ.
EPTQ is a post-training quantization method for CV networks.
It uses the network's loss function Hessian trace to perform an adaptive optimization for the rounding error of the 
quantized parameters, without needing a labeled dataset.

<p align="center">
  <img src="images/EPTQ.svg" width="800">
</p>

## Models

We provide a large set of pre-trained models. 
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

**Note:** Our implementation of EPTQ is based on the tools provided by 
[**Model-Compression-Toolkit (MCT)**](https://github.com/sony/model_optimization) version 1.9.

## Usage

### Single precision quantization
`python main.py -m resnet18 --weights_nbits 4 --activation_nbits 8 --train_data_path <path_to_training_dataset> 
--val_data_path <path_to_validation_dataset>`

This example would execute EPTQ via MCT to quantize ResNet18 with 4 bits for weights and 8 bits for activation, 
using 1,024 training samples.

For faster execution you can reduce the number of optimization steps, using the flag 
`--eptq_num_calibration_iter` (80K by default). This might result in a small reduction in the final accuracy results.

### Mixed precision quantization
We also enable mixed precision quantization, in which different weights tensors can be quantized 
with other bit-width, under pre-defined weights memory restrictions (total weights compression).

`python main.py -m regnetx_006 --mixed_precision --weights_cr 8 --train_data_path <path_to_training_dataset> 
--val_data_path <path_to_validation_dataset>`

This example would execute EPTQ via MCT to quantize RegNet-600MF with the bit-width options of 2, 4, and 8 bits for 
both weights and activations, while compressing the **total** memory of the model by a factor of 8.

