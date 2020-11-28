import numpy as np


def tanh(Z):
    return np.tanh(Z).astype(np.float32)

def d_tanh(Z):
    dZ = 1 - np.tanh(Z)**2
    return dZ