import numpy as np
from matplotlib import pyplot as plt

BRECQ = "BRECQ"
HAWQV3 = "HAWQ-V3"
GumbelRounding = "Gumbel-Rounding"
GumbelRoundingMPOnly = "Gumbel-Rounding"
mbv2 = {BRECQ: ((),)}
resnet18 = {HAWQV3: ((44.6, 71.47), [(9.9, 71.20), (7.9, 70.50), (7.3, 70.01)]),
            BRECQ: (
                (44.6, 71.08), [(5.5, 70.53), (5.0, 70.13), (4.5, 69.53), (4.0, 68.82), (3.5, 67.99), (3.0, 66.09)])}


def plot_accuracy(in_dict: dict, delta=False, precision=0.01):
    div = 1 / precision
    for k, v in in_dict.items():  # Loop over methods
        baseline = v[0]
        res = np.asarray([[size, acc] for size, acc in v[1]])
        res[:, 0] = res[:, 0]
        if delta:
            res[:, 1] = baseline[1] - res[:, 1]
        plt.plot(res[:, 0], res[:, 1], "-x", label=k)
        for a, b in zip(res[:, 0], res[:, 1]):
            plt.text(a, b, str(np.round(b * div) / div))
    plt.grid()
    plt.legend()
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Delta Accuracy")
    plt.show()
    # print("a")


plot_accuracy(resnet18, delta=True)
