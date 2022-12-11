import numpy as np


def log_norm(in_x):
    s = np.log10(in_x)
    return (s - np.min(s)) / (np.max(s) - np.min(s))
