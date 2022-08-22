import numpy as np
from matplotlib import pyplot as plt

BRECQ = "BRECQ"
HAWQV3 = "HAWQ-V3"
GumbelRounding = "Gumbel-Rounding"
GumbelRoundingMPOnly = "Gumbel-Rounding (Only MP)"
HMQ = "HMQ"
# mbv2 = {BRECQ: ((),)}
max_activation_tensor_resnet = 3.04
max_activation_tensor_mbv2 = 4.59
resnet18 = {
    HAWQV3: ((44.6, 71.47), [(9.9, 71.20), (7.9, 70.50), (7.3, 70.01)]),
    BRECQ: (
        (44.6, 71.08), [(5.5, 70.53), (5.0, 70.13), (4.5, 69.53), (4.0, 68.82), (3.5, 67.99), (3.0, 66.09)]),
    # GumbelRoundingMPOnly: (
    #     (44.6, 69.7), [(5.5, 67.728), (5.0, 64.432), (4.5, 61.718), (4.0, 56.39), (3.5, 50.464), (3.0, 14.388)]),
    GumbelRounding: (
        (44.6, 69.7), [(5.5, 69.204), (5.0, 68.17), (4.5, 68.642), (4.0, 68.359), (3.5, 66.839), (3.0, 66.014)])}

mbv2 = {BRECQ: ((13.23, 72.49), [(13.23 / 8, 71.39), (1.5, 70.28), (1.3, 68.99), (1.1, 67.4), (0.9, 63.73)]),
        GumbelRounding: (
            (13.23, 72.9), [(13.23 / 8, 71.725), (13.23 / 8.8, 71.613), (13.23 / 9.8, 70.477), (13.23 / 11, 69.183),
                            (13.23 / 12.5, 67.729), (13.23 / 14.66, 65.418)]),
        # HMQ: (
        #     (13.23, 71.88), [(13.23 / 7.7, 71.4), (13.23 / 9.71, 70.12), (13.23 / 14.4, 65.7)])
        }

print("a")


def plot_accuracy(in_dict: dict, delta=False, precision=0.01, shift_act=0.0):
    div = 1 / precision
    for k, v in in_dict.items():  # Loop over methods
        baseline = v[0]
        res = np.asarray([[size, acc] for size, acc in v[1]])
        res[:, 0] = res[:, 0]
        if delta:
            res[:, 1] = baseline[1] - res[:, 1]
        plt.plot(res[:, 0] + shift_act, res[:, 1], "-x", label=k)
        for a, b in zip(res[:, 0], res[:, 1]):
            plt.text(a + shift_act, b, str(np.round(b * div) / div))
    plt.grid()
    # plt.plot(res[:, 0], np.ones(len(res[:, 0])) * baseline[-1], label="Float")
    plt.legend()
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Accuracy")
    plt.show()
    # print("a")


plot_accuracy(mbv2, delta=True, shift_act=max_activation_tensor_mbv2 / 4)
