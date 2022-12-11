import pickle
import torch
from matplotlib import pyplot as plt
import numpy as np

from analysis.hessian_analysis.model_factory import get_model, Model


def log_norm(in_x):
    s = np.log10(in_x)
    return (s - np.min(s)) / (np.max(s) - np.min(s))


model = Model.ResNet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_list = []
data_highlight_list = []
point_high = []
for m, pretrained in enumerate([True]):
    net, name = get_model(model, pretrained, device)
    weights_type = "pretrained" if pretrained else "random"
    # Open the file in binary mode
    with open(f'../{name}.pkl', 'rb') as file:
        data_pickle = pickle.load(file)  # Call load method to deserialze
    lfha = data_pickle[0]
    trace_h = data_pickle[1]
    key_list = lfha.keys()
    print(key_list)
    key_to_highlight = ["bn1", *[key for key in lfha.keys() if "add" in key], "fc"]

    x_data = [2 * np.mean(lfha[key]) / 1000 for key in key_list]
    x_true = [np.mean(trace_h[key]) for key in key_list]


    def preprocess(in_x):
        _x_log = np.log10(in_x)
        return (_x_log - np.max(_x_log)) / (np.max(_x_log) - np.min(_x_log))


    # if m == 0:
    #     plt.semilogy([i for i, key in enumerate(key_list) if key in key_to_highlight],
    #                  [2 * np.mean(batch_jac_norm[key]) / 1000 for key in key_list if key in key_to_highlight],
    #                  "o", label="Block/Single Layer")
    data_highlight_list.append([2 * np.mean(lfha[key]) / 1000 for key in key_list if key in key_to_highlight])
    data_highlight_list.append([np.mean(trace_h[key]) for key in key_list if key in key_to_highlight])
    data_list.append(x_data)
    data_list.append(x_true)
    point_high.append([i for i, key in enumerate(key_list) if key in key_to_highlight])
    point_high.append([i for i, key in enumerate(key_list) if key in key_to_highlight])
    # plt.
    plt.subplot(1, 2, 1)
    plt.semilogy(x_data, "--",
                 label=f"LFHA-{weights_type}")

    plt.xlabel("Layer Index")
    plt.ylabel(r"$\mathbb{E}[\mathrm{Tr}(\mathbf{H})]$")
    plt.subplot(1, 2, 2)
    plt.plot(log_norm(x_data), "--",
             label=f"LFHA-{weights_type}")

    plt.xlabel("Layer Index")
    plt.ylabel(r"$\mathbb{E}[\mathrm{Tr}(\mathbf{H})]$")
plt.subplot(1, 2, 1)
label_mark = True
key2stage = ["layer1.0.bn1", "layer2.0.bn1", "layer3.0.bn1", 'layer4.0.bn1']
for i, key in enumerate(key_list):
    if key in key2stage:
        plt.plot([i, i], [np.min(np.stack(data_list)), np.max(np.stack(data_list))], "--", color="red",
                 label="Stage Crossing" if label_mark else None)
        label_mark = False

# x2 = [i for i, key in enumerate(key_list) if key in key_to_highlight]
plt.semilogy(np.stack(point_high).flatten(),
             # [2 * np.mean(batch_jac_norm[key]) / 1000 for key in key_list if key in key_to_highlight],
             np.stack(data_highlight_list).flatten(),
             "o", label="Block/Single Layer")

plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()
