import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from analysis.hessian_analysis.model_factory import get_model, Model
from matplotlib import pyplot as plt
from analysis.hessian_analysis.utils import log_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

point2mark_x = []
point2mark_y = []

for m, (model, pretrained) in enumerate(
        [(Model.ResNet18, True), (Model.ResNet18, False)]):
    net, name = get_model(model, pretrained, device)
    # weights_type = "pretrained" if pretrained else "random"
    # Open the file in binary mode
    with open(f'/data/projects/swat/users/haih/gumbel-rounding/{name}.pkl', 'rb') as file:
        data_pickle = pickle.load(file)  # Call load method to deserialze
    lfha = data_pickle[0]

    key_list = lfha.keys()

    key_to_highlight = ["bn1", *[key for key in lfha.keys() if "add" in key], "fc"]
    x_data = log_norm([2 * np.mean(lfha[key]) / 1000 for key in key_list])
    for i, k in enumerate(key_list):
        if k in key_to_highlight:
            point2mark_x.append(i)
            point2mark_y.append(x_data[i])

    plt.plot(x_data, "--",
             label=f"{model}" if pretrained else "Random")

    plt.xlabel("Layer Index")
    plt.ylabel(r"$\mathrm{}\mathbb{E}[\mathrm{Tr}(\mathbf{H})]$")

plt.plot(np.stack(point2mark_x).flatten(),
         np.stack(point2mark_y).flatten(),
         "o", label="Block/Single Layer")
plt.legend()
plt.grid()
plt.xlabel("Layer Index")
# plt.xlabel("Layer Index")
plt.show()
