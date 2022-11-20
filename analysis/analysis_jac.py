import torch
from torch import nn
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models.resnet import ResNet18_Weights, Optional, handle_legacy_interface, Any, ResNet, _resnet, \
    Callable, conv3x3, Tensor


class Add(nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.add = Add()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        # out = self.relu2(out)

        return out


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'


class Net(nn.Module):
    def __init__(self, n, m):
        super(Net, self).__init__()
        self.linear = nn.Linear(n, m)

    def forward(self, input):
        x = self.linear(input)
        return x


def model_register_hook(in_net, list2append):
    def get_activation(in_name, is_add):
        def hook(model, input, output):
            list2append.update({in_name: (output, is_add)})

        return hook

    for name, module in in_net.named_modules():
        if not isinstance(module, nn.Sequential) and (
                isinstance(module, nn.BatchNorm2d) or isinstance(module, Add) or isinstance(module,
                                                                                            nn.Linear)):  # or isinstance(module, nn.ReLU)):
            module.register_forward_hook(get_activation(name, isinstance(module, Add)))


def update_dict(agg_dict, in_dict):
    if len(agg_dict) == 0:
        return {k: [v] for k, v in in_dict.items()}
    else:
        for k, v in in_dict.items():
            agg_dict[k].append(v)
        return agg_dict


def compute_hessian_trace(in_net, x, y, in_criterion, in_n_iter, in_device):
    activations = {}
    model_register_hook(in_net, activations)
    res_image = {}
    x = x.to(in_device)
    y = y.to(in_device)
    for i in tqdm(range(x.shape[0])):
        print(f"Image {i} out of total {x.shape[0]} in batch")
        activations.clear()
        xi = x[i, :].unsqueeze(0)
        yi = y[i].unsqueeze(0)
        output = in_net(xi)
        loss = in_criterion(output, torch.nn.functional.one_hot(yi, 1000).float())
        res_tensor = {}
        for name, (activation_tensor, is_add) in activations.items():  # for each layer's output
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
            res_tensor.update({name: trace_res.item()})
        # res_image.append(res_tensor)
        res_image = update_dict(res_image, res_tensor)
    return res_image


def compute_jacobian_trace_approx(in_net, in_n_iter, in_input_tensors, in_device, updated=False):
    activations = {}
    model_register_hook(in_net, activations)
    # batch_jac_trace = []
    # batch_jac_norm = []
    per_layer_image_approx = {}
    for i in tqdm(range(len(in_input_tensors))):  # for each image in batch
        activations.clear()
        x = in_input_tensors[i]
        output = in_net(x.to(in_device))
        layers_jac_norm = {}
        add_or_conv = []
        for name, (activation_tensor, is_add) in activations.items():  # for each layer's output
            add_or_conv.append(is_add)
            jac_trace_approx = []
            for k in range(in_n_iter):  # iterations over random vectors
                v = torch.randn(output.shape, device=in_device)
                out_v = torch.mean(torch.sum(v * output, dim=-1))

                jac_v = autograd.grad(outputs=out_v,
                                      inputs=activation_tensor,
                                      retain_graph=True)[0]
                jac_v = torch.reshape(jac_v, [jac_v.shape[0], -1])
                if updated:
                    u = torch.randn(output.shape, device=in_device)
                    out_u = torch.mean(torch.sum(u * output, dim=-1))
                    jac_u = autograd.grad(outputs=out_u,
                                          inputs=activation_tensor,
                                          retain_graph=True)[0]
                    jac_u = torch.reshape(jac_u, [jac_v.shape[0], -1])
                    scale = torch.sum(v * u)
                    jac_trace = scale * torch.mean(torch.sum(jac_v * jac_u))
                else:
                    jac_trace = torch.mean(torch.sum(torch.pow(jac_v, 2.0)))
                c = activation_tensor.shape[1]
                h = w = 1
                if len(activation_tensor.shape) == 4:
                    h = activation_tensor.shape[2]
                    w = activation_tensor.shape[3]
                jac_trace_approx.append(jac_trace.item())
            # per_layer_trace_approx.append(jac_trace_approx)
            layers_jac_norm.update({name: np.mean(jac_trace_approx)})
        per_layer_image_approx = update_dict(per_layer_image_approx, layers_jac_norm)
        # batch_jac_norm.append(layers_jac_norm)
        # print(f"Current Jacobian approximation per layer is: \n {np.mean(batch_jac_norm, axis=0)}")

    return per_layer_image_approx


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
    n_iter = 50
    batch_size = 2
    # net = models.mobilenet_v2(pretrained=True).to(device)
    net = resnet18(pretrained=True).to(device)
    # net = models.regnet_x_400mf(pretrained=True).to(device)
    net = net.eval()
    dl = get_validation_loader(batch_size)

    x, y = next(iter(dl))

    # results = compute_hessian_trace(net, x, y, nn.MSELoss(), n_iter, device)
    input_tensors = [x[i - 1:i, :, :, :] for i in range(1, x.shape[0] + 1)]
    for t in input_tensors:
        t.requires_grad_()
    batch_jac_norm = compute_jacobian_trace_approx(in_net=net, in_n_iter=n_iter,
                                                   in_input_tensors=input_tensors,
                                                   in_device=device)

    key_list = batch_jac_norm.keys()
    print(key_list)
    key_to_highlight = ["bn1", "layer1.0.add", "layer1.1.add", 'layer2.0.add', 'layer2.1.add', 'layer3.0.add',
                        'layer3.1.add', 'layer4.0.add', 'layer4.1.add', 'fc']
    # trace_mean = np.mean(batch_jac_norm, axis=0)
    # [np.mean(results[key]) for key in key_list]
    # plt.semilogy([np.mean(results[key]) for key in key_list], label="Hessian")
    plt.semilogy([i for i, key in enumerate(key_list) if key in key_to_highlight],
                 [2 * np.mean(batch_jac_norm[key]) / 1000 for key in key_list if key in key_to_highlight],
                 "o", label="Block")

    plt.semilogy([2 * np.mean(batch_jac_norm[key]) / 1000 for key in key_list], "--x", label="Label Free Approximation")
    plt.xlabel("Layer Index")
    plt.ylabel(r"$\mathbb{E}[\mathrm{Tr}(\mathbf{H})]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_hawq_resnet18.svg")
    # plt.cla()
    # plt.clf()
    plt.show()

    # running_mean = np.mean(np.sqrt(
    #     np.cumsum(jac_norm_approx_array, axis=-1) / np.reshape(np.cumsum(np.ones(jac_norm_approx_array.shape[-1])),
    #                                                            [1, 1, -1])), axis=0)
    # for i_layer in range(running_mean.shape[0]):
    #     plt.plot(running_mean[i_layer, :], label="Result Per Iteration")
    #     plt.plot(running_mean[i_layer, -1] * np.ones(running_mean.shape[-1]), label="Final Result")
    #     plt.xlabel("Iteration")
    #     plt.ylabel(r"$\mathbb{E}[||\mathbf{J}||_F]$")
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(f"trace_convergence_{i_layer}.svg")
    #     plt.cla()
    #     plt.clf()
