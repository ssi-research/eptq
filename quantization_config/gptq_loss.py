from typing import List

import numpy as np
import tensorflow as tf


def mse_loss_per_tensor(y: tf.Tensor, x: tf.Tensor, normalized: bool = True, p: int = 2) -> tf.Tensor:
    """
    Compute the MSE of two tensors.
    Args:
        y: First tensor.
        x: Second tensor.
        normalized: either return normalized MSE (default), or MSE
        p: either return normalized MSE (default), or MSE

    Returns:
        The MSE of two tensors.
    """
    _loss = tf.reduce_mean(tf.pow(tf.abs(y - x), p))
    return _loss / tf.reduce_mean(tf.pow(tf.abs(x), p)) if normalized else _loss


def activation_mse(flp_act_list, fxp_act_list, p_vector=None, weights_for_average_loss=None):
    loss_values_list = []
    bias_loss_list = []
    for i, (flp_act, fxp_act) in enumerate(zip(flp_act_list, fxp_act_list)):
        if p_vector is None:
            p = 2.0
        else:
            p = p_vector[i]
        point_loss = mse_loss_per_tensor(fxp_act, flp_act, p=p)
        delta = flp_act - fxp_act
        m = len(delta.shape)

        bias_loss = tf.reduce_mean(tf.square(tf.reduce_sum(delta, axis=tuple(range(1, m)))))  # Tensor bias
        if m == 4:
            bias_loss = bias_loss + tf.reduce_mean(tf.square(tf.reduce_sum(delta, axis=(1, 2))))  # Channel bias

        loss_values_list.append(point_loss)
        bias_loss_list.append(bias_loss)
    if weights_for_average_loss is not None:
        return tf.reduce_sum(weights_for_average_loss * tf.stack(loss_values_list)), \
               tf.reduce_mean(tf.stack(bias_loss_list))
    else:
        return tf.reduce_sum(tf.stack(loss_values_list)), tf.reduce_mean(tf.stack(bias_loss_list))
    # return tf.reduce_sum(tf.stack(loss_values_list)), tf.reduce_mean(tf.stack(bias_loss_list))


class GPTQMultipleTensorsLoss:
    def __init__(self):
        self.alpha = None

    # def upload_loss_alpha(self, *args):
    #     if self.alpha is None:
    #         self.alpha = np.zeros([len(args)])
    #     for i, li in enumerate(args):  # alpha EMA
    #         self.alpha[i] += (1 - self.beta) * (np.log(li + self.eps) - self.alpha[i])
    #
    # def weight_sum_loss(self, *args):
    #     total_loss = 0
    #     for i, t in enumerate(args):
    #         alpha = self.alpha[i] / (1 - self.beta ** self.i)  # Bias correction of EMA
    #         total_loss = total_loss + np.exp(-alpha) * t
    #     return total_loss

    def __call__(self, fxp_act_list: List[tf.Tensor], flp_act_list: List[tf.Tensor],
                 fxp_w_list: List[List[tf.Tensor]],
                 flp_w_list: List[List[tf.Tensor]],
                 act_bn_mean: List,
                 act_bn_std: List,
                 weights_for_average_loss: List,
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
        # p_vector = [1 + 0.9 * tf.sigmoid(tf.reduce_mean(gamma) - 1) if gamma is not None else 2.0 for gamma in
        #             act_bn_std]
        p_vector = None

        loss_act, loss_activation_bias = activation_mse(flp_act_list, fxp_act_list, p_vector=p_vector,
                                                        weights_for_average_loss=weights_for_average_loss)

        return loss_act
