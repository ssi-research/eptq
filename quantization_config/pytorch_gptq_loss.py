from typing import List
import torch

def mse_loss_per_tensor(y: torch.Tensor, x: torch.Tensor, normalized: bool = False, p: int = 2) -> torch.Tensor:
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
    _loss = torch.mean(torch.pow(torch.abs(y - x), p))
    return _loss / torch.mean(torch.pow(torch.abs(x), p)) if normalized else _loss


def activation_mse(flp_act_list, fxp_act_list, p_vector=None, weights_for_average_loss=None, norm_loss=True):
    loss_values_list = []
    bias_loss_list = []
    for i, (flp_act, fxp_act) in enumerate(zip(flp_act_list, fxp_act_list)):
        if p_vector is None:
            p = 2.0
        else:
            p = p_vector[i]
        point_loss = mse_loss_per_tensor(fxp_act, flp_act, p=p, normalized=norm_loss)
        delta = flp_act - fxp_act
        m = len(delta.shape)

        bias_loss = torch.mean(torch.square(torch.sum(delta, dim=tuple(range(1, m)))))  # Tensor bias
        if m == 4:
            bias_loss = bias_loss + torch.mean(torch.square(torch.sum(delta, dim=(1, 2))))  # Channel bias

        loss_values_list.append(point_loss)
        bias_loss_list.append(bias_loss)
    if weights_for_average_loss is not None:
        return torch.sum(weights_for_average_loss * torch.stack(loss_values_list)), \
               torch.sum(weights_for_average_loss * torch.stack(bias_loss_list))
    else:
        return torch.mean(torch.stack(loss_values_list)), torch.mean(torch.stack(bias_loss_list))


class GPTQMultipleTensorsLoss:
    def __init__(self, norm_loss: bool = True):
        self.alpha = None
        self.norm_loss = norm_loss

    def __call__(self,
                 fxp_act_list: List[torch.Tensor],
                 flp_act_list: List[torch.Tensor],
                 fxp_w_list: List[List[torch.Tensor]],
                 flp_w_list: List[List[torch.Tensor]],
                 act_bn_mean: List,
                 act_bn_std: List,
                 weights_for_average_loss: List,
                 ) -> torch.Tensor:
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
        p_vector = None
        loss_act, loss_activation_bias = activation_mse(flp_act_list, fxp_act_list, p_vector=p_vector,
                                                        weights_for_average_loss=weights_for_average_loss,
                                                        norm_loss=self.norm_loss)
        return loss_act
