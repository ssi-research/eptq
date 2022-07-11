from typing import List

import model_compression_toolkit as mct
from model_compression_toolkit import RoundingType
import tensorflow as tf

from utils.radam_optimizer import RAdam


def build_gptq_config(args):
    rounding_type = RoundingType.STE if args.ste_rounding else RoundingType.GumbelRounding

    return mct.GradientPTQConfig(n_iter=args.gptq_num_calibration_iter,
                                 optimizer=RAdam(learning_rate=args.lr,
                                                 total_steps=args.gptq_num_calibration_iter,
                                                 warmup_proportion=0.2),
                                 loss=MultipleTensorsMse(weights_loss_factor=0.1,
                                                         bn_loss_factor=0.0),
                                 log_function=None,  # TODO: add logging function when adding WANDB support
                                 train_bias=args.bias_learning,
                                 quantization_parameters_learning=args.quantization_parameters_learning_weights,
                                 temperature_learning=args.temperature_learning,
                                 sam_optimization=args.sam_optimization,
                                 rounding_type=rounding_type,
                                 rho=args.rho,
                                 gumbel_entropy_regularization=args.gamma_temperature)


class MultipleTensorsMse:
    def __init__(self, weights_loss_factor: float = 0.0, bn_loss_factor: float = 0.0):
        self.weights_loss_factor = weights_loss_factor
        self.bn_loss_factor = bn_loss_factor

    def __call__(self, fxp_act_list: List[tf.Tensor], flp_act_list: List[tf.Tensor],
                 fxp_w_list: List[List[tf.Tensor]],
                 flp_w_list: List[List[tf.Tensor]],
                 act_bn_mean: List,
                 act_bn_std: List,
                 ) -> tf.Tensor:
        """
        Compute mse between two lists of tensors. The returned loss is an
        average of mse loss per layer

        Args:
            fxp_act_list: First list of tensors: the activations of the quantized model
            flp_act_list: Second list of tensors: the activations of the float model
            fxp_w_list: list of trainable weights - quantized
            flp_w_list: list of trainable weights - not quantized
            act_bn_mean: list of prior activations mean collected from batch normalization. None is there's no info
            act_bn_std: list of prior activations std collected from batch normalization. None is there's no info

        Returns:
            List of cosine similarities.
        """

        loss_values_list = []
        for i, (flp_act, fxp_act) in enumerate(zip(flp_act_list, fxp_act_list)):
            point_loss = mse_loss_per_tensor(fxp_act, flp_act)
            loss_values_list.append(point_loss)
        loss = tf.reduce_mean(tf.stack(loss_values_list))

        if self.weights_loss_factor > 0:
            fxp_w_list = [w for layer in fxp_w_list for w in layer]
            flp_w_list = [w for layer in flp_w_list for w in layer]
            loss_values_list = []
            for i, (wq, w) in enumerate(zip(fxp_w_list, flp_w_list)):
                if len(w.shape) == 2:
                    # Dense layer needs to reshape to (H, W, Cin, Cout): (Cin, Cout) --> (1, 1, Cin, Cout)
                    w = tf.expand_dims(tf.expand_dims(w, 0), 0)
                    wq = tf.expand_dims(tf.expand_dims(wq, 0), 0)
                dw = wq - w
                loss_values_list.append(tf.reduce_mean(tf.reduce_sum(tf.square(dw), axis=[0, 1, 2]) +
                                                       tf.reduce_sum(tf.square(tf.reduce_sum(dw, axis=[0, 1])), axis=0) +
                                                       tf.square(tf.reduce_sum(dw, axis=[0, 1, 2]))))
            weights_loss = tf.reduce_mean(tf.stack(loss_values_list))
            loss = loss + self.weights_loss_factor*weights_loss

        if self.bn_loss_factor > 0:
            loss_values_list = []
            for i, (act_mean, act_std, fxp_act) in enumerate(zip(act_bn_mean, act_bn_std, fxp_act_list)):
                if act_mean is not None and act_std is not None:
                    non_channel_axis = list(range(len(fxp_act.shape)-1))
                    _mean = tf.reduce_mean(fxp_act, axis=non_channel_axis)
                    _var = tf.reduce_mean(tf.square(fxp_act - tf.reduce_mean(fxp_act, axis=non_channel_axis, keepdims=True)),
                                          axis=non_channel_axis)
                    mean_loss = mse_loss_per_tensor(_mean, act_mean)
                    std_loss = mse_loss_per_tensor(_var, tf.square(act_std))
                    loss_values_list.append(mean_loss + std_loss)
                    # loss_values_list.append(mean_loss)
            bn_loss = tf.reduce_mean(tf.stack(loss_values_list))
            loss = loss + self.bn_loss_factor*bn_loss

        return loss


def mse_loss_per_tensor(y: tf.Tensor, x: tf.Tensor, normalized: bool = True) -> tf.Tensor:
    """
    Compute the MSE of two tensors.
    Args:
        y: First tensor.
        x: Second tensor.
        normalized: either return normalized MSE (default), or MSE

    Returns:
        The MSE of two tensors.
    """
    _loss = tf.reduce_mean(tf.pow(y - x, 2.0))
    return _loss / tf.reduce_mean(tf.pow(x, 2.0)) if normalized else _loss