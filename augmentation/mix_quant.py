import torch


class MixedQuantized(object):
    def __init__(self, alpha):
        self.alpha = alpha
        if self.alpha > 0:
            # TODO: make it an argument
            self.dist = torch.distributions.beta.Beta(alpha, alpha)
        else:
            self.dist = None

    def __call__(self, x):
        batch_size = x.size()[0]
        if self.dist is not None:
            gamma = self.dist.sample(sample_shape=[batch_size, 1, 1, 1])
        else:
            gamma = 1

        index = torch.randperm(batch_size).to(x.device)
        mixed_x = gamma * x + (1 - gamma) * x[index, :]
        return mixed_x
        # return torch.round(255 * mixed_x).float() / 255  # Apply quantization and return to float.
