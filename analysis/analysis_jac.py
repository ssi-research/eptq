import torch
from torch import nn
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm

VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'


class Net(nn.Module):
    def __init__(self, n, m):
        super(Net, self).__init__()
        self.linear = nn.Linear(n, m)

    def forward(self, input):
        x = self.linear(input)
        return x


def model_register_hook(in_net, list2append):
    def get_activation():
        def hook(model, input, output):
            list2append.append(output)

        return hook

    for module in list(in_net.modules())[1:]:
        if not isinstance(module, nn.Sequential) and (
                isinstance(module, nn.BatchNorm2d)):  # or isinstance(module, nn.ReLU)):
            module.register_forward_hook(get_activation())


def compute_hessian_trace(in_net, in_dataloader, in_criterion, in_n_iter, in_device):
    activations = []
    model_register_hook(in_net, activations)
    res_image = []
    for x, y in in_dataloader:  # for each image in batch
        x = x.to(in_device)
        y = y.to(in_device)
        for i in tqdm(range(x.shape[0])):
            print(f"Image {i} out of total {x.shape[0]} in batch")
            activations.clear()
            xi = x[i, :].unsqueeze(0)
            yi = y[i].unsqueeze(0)
            output = in_net(xi)
            loss = in_criterion(output, yi)
            res_tensor = []
            for j, activation_tensor in enumerate(activations):  # for each layer's output
                grad = autograd.grad(outputs=loss,
                                     inputs=activation_tensor,
                                     retain_graph=True, create_graph=True)[0]
                grad = grad.reshape([-1])
                acc = 0
                for k in range(in_n_iter):  # iterations over random vectors
                    v = torch.randn(grad.shape, device=in_device)
                    gv = torch.sum(grad * v)
                    hv = autograd.grad(outputs=gv,
                                       inputs=activation_tensor,
                                       retain_graph=True)[0]
                    acc += torch.sum(v * hv.reshape([-1]))
                trace_res = acc / in_n_iter
                res_tensor.append(trace_res.item())
            res_image.append(res_tensor)
        break
    return np.asarray(res_image)


def compute_jacobian_trace_approx(in_net, in_n_iter, in_input_tensors, in_device):
    activations = []
    model_register_hook(in_net, activations)
    # batch_jac_trace = []
    batch_jac_norm = []
    per_layer_image_approx = []
    for i in tqdm(range(len(in_input_tensors))):  # for each image in batch
        # print(f"Image {i} out of total {len(in_input_tensors)} in batch")
        activations.clear()
        x = in_input_tensors[i]
        output = in_net(x.to(in_device))
        # layers_jac_trace = []
        layers_jac_norm = []
        per_layer_trace_approx = []
        for j, activation_tensor in enumerate(activations):  # for each layer's output
            jac_trace_approx = []
            for k in range(in_n_iter):  # iterations over random vectors
                v = torch.randn(output.shape, device=in_device)
                out_v = torch.mean(torch.sum(v * output, dim=-1))

                jac_v = autograd.grad(outputs=out_v,
                                      inputs=activation_tensor,
                                      retain_graph=True)[0]

                jac_v = torch.reshape(jac_v, [jac_v.shape[0], -1])
                jac_trace = torch.mean(torch.sum(torch.pow(jac_v, 2.0)))

                jac_trace_approx.append(jac_trace.item())
            per_layer_trace_approx.append(jac_trace_approx)
            # layers_jac_trace.append(np.mean(jac_trace_approx))
            layers_jac_norm.append(np.sqrt(np.mean(jac_trace_approx)))
        per_layer_image_approx.append(per_layer_trace_approx)
        # batch_jac_trace.append(layers_jac_trace)
        batch_jac_norm.append(layers_jac_norm)
        print(f"Current Jacobian approximation per layer is: \n {np.mean(batch_jac_norm, axis=0)}")

    return np.asarray(batch_jac_norm), np.asarray(per_layer_image_approx)


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_iter = 200
    batch_size = 64
    net = models.mobilenet_v2(pretrained=True).to(device)
    net = net.eval()
    dl = get_validation_loader(batch_size)
    samples = next(iter(dl))[0]
    input_tensors = [samples[i - 1:i, :, :, :] for i in range(1, samples.shape[0] + 1)]
    for t in input_tensors:
        t.requires_grad_()
    results = compute_hessian_trace(net, dl, nn.CrossEntropyLoss(), 10, device)
    batch_jac_norm, jac_norm_approx_array = compute_jacobian_trace_approx(in_net=net, in_n_iter=n_iter,
                                                                          in_input_tensors=input_tensors,
                                                                          in_device=device)
    trace_mean = np.mean(batch_jac_norm, axis=0)
    # plt.subplot(1, 2, 1)
    plt.plot(np.sqrt(trace_mean), label="trace")
    plt.title("JTJ Trace")
    plt.xlabel("Layer Index")
    plt.ylabel(r"$\mathbb{E}[||\mathbf{J}||_F]$")
    plt.grid()
    plt.savefig("ejn.svg")
    plt.cla()
    plt.clf()

    # plt.show()
    # plt.subplot(1, 2, 2)
    plt.plot(np.mean(results, axis=0), label="Hessian")
    plt.xlabel("Layer Index")
    plt.ylabel(r"$\mathbb{E}[\mathrm{Tr}(\mathbf{H})]$")
    plt.title("Hessian Trace")
    plt.grid()
    plt.savefig("hawq.svg")
    plt.cla()
    plt.clf()

    running_mean = np.mean(np.sqrt(
        np.cumsum(jac_norm_approx_array, axis=-1) / np.reshape(np.cumsum(np.ones(jac_norm_approx_array.shape[-1])),
                                                               [1, 1, -1])), axis=0)
    for i_layer in range(running_mean.shape[0]):
        plt.plot(running_mean[i_layer, :], label="Result Per Iteration")
        plt.plot(running_mean[i_layer, -1] * np.ones(running_mean.shape[-1]), label="Final Result")
        plt.xlabel("Iteration")
        plt.ylabel(r"$\mathbb{E}[||\mathbf{J}||_F]$")
        plt.legend()
        plt.grid()
        plt.savefig(f"trace_convergence_{i_layer}.svg")
        plt.cla()
        plt.clf()
