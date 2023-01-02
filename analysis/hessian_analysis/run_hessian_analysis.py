import torch
from torch import nn
import torch.autograd as autograd
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from analysis.hessian_analysis.custom_layers import Add
from analysis.hessian_analysis.model_factory import get_model, Model
import pickle

VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'


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


def compute_hessian_trace(in_net, in_x, in_y, in_criterion, in_n_iter, in_device, minimal_iteration=10, eps=1e-3,
                          in_one_hot=True):
    activations = {}
    model_register_hook(in_net, activations)
    res_image = {}
    in_x = in_x.to(in_device)
    in_y = in_y.to(in_device)
    for i in tqdm(range(in_x.shape[0])):
        print(f"Image {i} out of total {in_x.shape[0]} in batch")
        activations.clear()
        xi = in_x[i, :].unsqueeze(0)
        yi = in_y[i].unsqueeze(0)
        output = in_net(xi)
        if in_one_hot:
            yi = torch.nn.functional.one_hot(yi, 1000).float().detach()

        loss = in_criterion(output, yi)
        res_tensor = {}
        for in_name, (activation_tensor, is_add) in activations.items():  # for each layer's output
            grad = autograd.grad(outputs=loss,
                                 inputs=activation_tensor,
                                 retain_graph=True, create_graph=True)[0]
            grad = grad.reshape([-1])
            acc = 0
            for k in range(in_n_iter):  # iterations over random vectors
                v = torch.randint_like(grad, high=2, device=device)
                v[v == 0] = -1

                gv = torch.sum(grad * v)
                hv = autograd.grad(outputs=gv,
                                   inputs=activation_tensor,
                                   retain_graph=True)[0]
                update_size = torch.sum(v * hv.reshape([-1]))
                n = k + 1
                if n > minimal_iteration:
                    delta = (acc + update_size) / n - acc / (n - 1)
                    if torch.abs(delta).item() < eps:
                        acc += update_size
                acc += update_size
            # print(delta, n)
            trace_res = acc / n
            res_tensor.update({in_name: trace_res.item()})
        res_image = update_dict(res_image, res_tensor)
    avg_samples_dict = {k: np.mean(v) for k, v in res_image.items()}
    return avg_samples_dict


def compute_jacobian_trace_approx(in_net, in_n_iter, in_input_tensors, in_device, min_iteration=10, eps=1e-3):
    activations = {}
    model_register_hook(in_net, activations)

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
                v = torch.randint_like(output, high=2, device=device)
                v[v == 0] = -1
                out_v = torch.mean(torch.sum(v * output, dim=-1))

                jac_v = autograd.grad(outputs=out_v,
                                      inputs=activation_tensor,
                                      retain_graph=True)[0]
                jac_v = torch.reshape(jac_v, [jac_v.shape[0], -1])
                jac_trace = torch.mean(torch.sum(torch.pow(jac_v, 2.0))).item()
                delta = np.mean([jac_trace, *jac_trace_approx]) - np.mean(jac_trace_approx)
                if k > min_iteration:
                    if np.abs(delta) < eps:
                        jac_trace_approx.append(jac_trace)
                        break
                jac_trace_approx.append(jac_trace)
            layers_jac_norm.update({name: np.mean(jac_trace_approx)})
        per_layer_image_approx = update_dict(per_layer_image_approx, layers_jac_norm)
    avg_samples_dict = {k: 2 * np.mean(v) / 1000 for k, v in per_layer_image_approx.items()}
    return avg_samples_dict


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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_iter = 500
    batch_size = 16
    run_ref = True
    for model in [Model.ResNet18, Model.ResNet50W1, Model.ResNet50W2, Model.MobileNetV2W1,
                  Model.MobileNetV2W2]:
        for pretrained in [True, False]:
            net, name = get_model(model, pretrained, device)

            net = net.eval()
            dl = get_validation_loader(batch_size)

            x, y = next(iter(dl))
            ref_data = None
            if run_ref and pretrained and model == Model.ResNet18:
                ref_data = {}
                for (criterion, one_hot) in [(nn.MSELoss(), True), (nn.CrossEntropyLoss(), False)]:
                    _ref_data = compute_hessian_trace(net, x, y, criterion, n_iter, device, in_one_hot=one_hot)
                    ref_data.update({criterion._get_name(): _ref_data})
            input_tensors = [x[i - 1:i, :, :, :] for i in range(1, x.shape[0] + 1)]
            for t in input_tensors:
                t.requires_grad_()
            batch_jac_norm = compute_jacobian_trace_approx(in_net=net, in_n_iter=n_iter,
                                                           in_input_tensors=input_tensors,
                                                           in_device=device)

            with open(f'../{name}.pkl', 'wb') as file:
                # A new file will be created
                pickle.dump((batch_jac_norm, ref_data), file)
