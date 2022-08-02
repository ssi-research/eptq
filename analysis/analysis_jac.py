import torch
from torch import nn
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt

VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'


class Net(nn.Module):
    def __init__(self, n, m):
        super(Net, self).__init__()
        self.linear = nn.Linear(n, m)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(1, m)

    def forward(self, input):
        x = self.linear(input)
        # x = self.relu(x)
        # x = self.linear2(x.T)
        return x


# n = 10  # Input size
# m = 5  # Output size
# batch_size = 1
# n_iter = 50
# num_samples = 32
#
# net = Net(n, m)
# # net = nn.Linear(n, m)
#
# jac = net.linear.weight
#
# jac_trace = torch.matmul(jac, jac.T).trace()
# print(jac_trace.item())
#
# # x = torch.randn(batch_size, n, requires_grad=True)
# # # x = x * torch.ones(x.shape, requires_grad=True)
# # y = net(x)
#
# samples_res_list = []
# for j in range(num_samples):
#     x = torch.randn(batch_size, n, requires_grad=True)
#     # x = x * torch.ones(x.shape, requires_grad=True)
#     y = net(x)
#     res_list = []
#     for i in range(n_iter):
#         v = torch.randn(batch_size, m)
#         # v = torch.randn(m)
#         l = torch.mean(torch.sum(v * y, dim=-1))
#
#         gradients = autograd.grad(outputs=l,
#                                   inputs=x,
#                                   retain_graph=True)[0]
#         # gradients = torch.reshape(gradients, [gradients.shape[0], 1, -1])
#         gradients = torch.reshape(gradients, [gradients.shape[0], -1])
#         a = torch.mean(torch.sum(torch.pow(gradients, 2.0)))
#         res_list.append(a.item())
#     samples_res_list.append(np.mean(res_list))
# print(np.mean(samples_res_list))


def compute_jacobian_trace_approx(net, n_iter, input_tensors):
    activations = []

    def get_activation():
        def hook(model, input, output):
            activations.append(output)

        return hook

    for module in list(net.modules())[1:]:
        if not isinstance(module, nn.Sequential) and (
                isinstance(module, nn.BatchNorm2d)):  # or isinstance(module, nn.ReLU)):
            module.register_forward_hook(get_activation())

    batch_jac_trace = []
    batch_jac_norm = []
    for i in range(len(input_tensors)):  # for each image in batch
        print(f"Image {i} out of total {len(input_tensors)} in batch")
        x = input_tensors[i]
        output = net(x)
        layers_jac_trace = []
        layers_jac_norm = []
        for j, activation_tensor in enumerate(activations):  # for each layer's output
            print(f"Computing Jacobian approximation for layer {j} out of {len(activations)}")
            jac_trace_approx = []
            # jac_norm_approx = []
            for k in range(n_iter):  # iterations over random vectors
                v = torch.randn(output.shape)
                out_v = torch.mean(torch.sum(v * output, dim=-1))

                jac_v = autograd.grad(outputs=out_v,
                                      inputs=activation_tensor,
                                      retain_graph=True)[0]

                jac_v = torch.reshape(jac_v, [jac_v.shape[0], -1])
                jac_trace = torch.mean(torch.sum(torch.pow(jac_v, 2.0)))

                jac_trace_approx.append(jac_trace.item())
            layers_jac_trace.append(np.mean(jac_trace_approx))
            layers_jac_norm.append(np.sqrt(np.mean(jac_trace_approx)))
        batch_jac_trace.append(layers_jac_trace)
        batch_jac_norm.append(layers_jac_norm)
        print(f"Current Jacobian approximation per layer is: \n {np.mean(batch_jac_trace, axis=0)}")
        activations = []

    return np.asarray(batch_jac_trace), np.asarray(batch_jac_norm), np.asarray(jac_trace_approx)


def get_default_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_validation_loader(batch_size=32):
    preprocess = get_default_preprocess()
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(VAL_DIR, preprocess),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)


def plot_jac_approx_per_layer(final_jac_approx):
    plt.plot(range(len(final_jac_approx)), final_jac_approx, "--o")
    plt.grid()
    plt.show()


if __name__ == '__main__':

    n_iter = 50
    batch_size = 16
    net = models.resnet18(pretrained=True)
    net = net.eval()

    samples = next(iter(get_validation_loader(batch_size)))[0]
    input_tensors = [samples[i - 1:i, :, :, :] for i in range(1, samples.shape[0] + 1)]
    for t in input_tensors:
        t.requires_grad_()

    batch_jac_trace, batch_jac_norm, jac_norm_approx_array = compute_jacobian_trace_approx(net=net, n_iter=n_iter,
                                                                                           input_tensors=input_tensors)

    running_mean = np.cumsum(jac_norm_approx_array) / np.cumsum(np.ones(len(jac_norm_approx_array)))
    trace_mean = np.mean(batch_jac_trace, axis=0)
    plt.plot(running_mean)
    plt.show()

    plt.plot(trace_mean , label="trace")
    plt.legend()
    plt.grid()
    plt.show()
    # print("a")
    # plt.subplot(2, 2, 1)
    # plt.plot(batch_jac_trace[0, :])

    # plt.plot(trace_mean)
    # plt.subplot(2, 2, 2)
    # plt.plot(batch_jac_norm[0, :])
    # norm_mean = np.mean(batch_jac_norm, axis=0)
    # plt.plot(norm_mean)
    # plt.subplot(2, 2, 3)
    # norm_mean_min = norm_mean - np.max(norm_mean)
    # plt.plot(norm_mean / np.sum(norm_mean), label="norm")
    # plt.plot(np.exp(norm_mean_min) / np.sum(np.exp(norm_mean_min)), label="norm-softmax")
    # batch_jac_trace = np.asarray(batch_jac_trace)

    # print(final_jac_approx)
    # plot_jac_approx_per_layer(final_jac_approx)
