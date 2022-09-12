import numpy as np
from matplotlib import pyplot as plt

BRECQ = "BRECQ"
HAWQV3 = r"HAWQ-V3$^*$"
GumbelRounding = "Gumbel-Rounding"
GumbelRoundingMPOnly = "Gumbel-Rounding (Only MP)"
HMQ = "HMQ"
OMPQ_QA = r"OMPQ$^*$"
OMPQ = r"OMPQ"
# mbv2 = {BRECQ: ((),)}
max_activation_tensor_resnet = 3.04
max_activation_tensor_mbv2 = 4.59
resnet18 = {
    HAWQV3: ((44.6, 71.47), [(9.9, 71.20), (7.9, 70.50), (7.3, 70.01)]),

    OMPQ: (
        (44.6, 71.08), [(5.5, 70.52), (5.0, 70.08), (4.5, 69.73), (4.0, 69.34), (3.5, 68.54), (3.0, 65.91)]),
    BRECQ: (
        (44.6, 71.08), [(5.5, 70.53), (5.0, 70.13), (4.5, 69.53), (4.0, 68.82), (3.5, 67.99), (3.0, 66.09)]),

    GumbelRounding: (
        (44.6, 69.7), [(5.5, 69.204), (5.0, 68.678), (4.5, 68.642), (4.0, 68.359), (3.5, 66.839), (3.0, 66.014)])}

mbv2 = {BRECQ: ([13.23, 72.49], [[13.23 / 8, 71.39], [1.5, 70.28], [1.3, 68.99], [1.1, 67.4], [0.9, 63.73]]),
        GumbelRoundingMPOnly: (
            (13.23 + max_activation_tensor_mbv2, 72.9),
            [(2919872 / 1e6, 72.098), (2738624 / 1e6, 71.99), (2594624 / 1e6, 71.882),
             (2459872 / 1e6, 70.656), (2335936 / 1e6, 70.168), (2124256 / 1e6, 63.232), (1907680 / 1e6, 47.696),
             (1699480 / 1e6, 31.834), (1495312 / 1e6, 0.3)]),
        OMPQ: ([13.23, 72.49], [[13.23 / 8, 71.60], [1.5, 71.27], [1.3, 69.51], [1.1, 67.65], [0.9, 63.81]]),
        HMQ: ([13.23 + max_activation_tensor_mbv2, 71.88], [[13.23 / 8 + max_activation_tensor_mbv2 / 8, 70.9]])
        # HMQ: (
        #     (13.23, 71.88), [(13.23 / 7.7, 71.4), (13.23 / 9.71, 70.12), (13.23 / 14.4, 65.7)])
        }


def add_memory(in_dict, method_name, act_size):
    in_dict[method_name][0][0] += act_size
    for j in in_dict[method_name][1]:
        j[0] += act_size / 4


add_memory(mbv2, BRECQ, max_activation_tensor_mbv2)
add_memory(mbv2, OMPQ, max_activation_tensor_mbv2)


# print("a")
# print("mbv2", [mbv2[BRECQ][0][0] / i[0] for i in mbv2[BRECQ][1]])
# print("mbv2-g", [mbv2[GumbelRounding][0][0] / i[0] for i in mbv2[GumbelRounding][1]])
# print("-" * 100)
# print("resnet18", [resnet18[BRECQ][0][0] / i[0] for i in resnet18[BRECQ][1]])
# print("resnet18-g", [resnet18[GumbelRounding][0][0] / i[0] for i in resnet18[GumbelRounding][1]])


def plot_accuracy(in_dict: dict, delta=False, precision=0.01, shift_act=0.0):
    div = 1 / precision
    for k, v in in_dict.items():  # Loop over methods
        baseline = v[0]
        res = np.asarray([[size, acc] for size, acc in v[1]])
        res[:, 0] = res[:, 0]
        if delta:
            res[:, 1] = 100 * res[:, 1] / baseline[1]
        if res.shape[0] == 1:
            plt.plot(res[:, 0] + shift_act, res[:, 1], "v", label=k)
        else:
            plt.plot(res[:, 0] + shift_act, res[:, 1], "-x", label=k)
        for a, b in zip(res[:, 0], res[:, 1]):
            plt.text(a + shift_act, b, str(np.round(b * div) / div))
    plt.grid()
    # plt.plot(res[:, 0], np.ones(len(res[:, 0])) * baseline[-1], label="Float")
    plt.legend()
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Accuracy Ratio")


plot_accuracy(resnet18, delta=True, shift_act=0.0)
plt.savefig("mbv2_w_mp.svg")
plt.show()
