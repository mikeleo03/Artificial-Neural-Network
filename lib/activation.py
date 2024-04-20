import numpy as np


# Fungsi aktivasi reLU
def relu(x, derivative=False):
    x = np.array(x, dtype=float)
    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)


# Fungsi aktivasi sigmoid
def sigmoid(x, derivative=False):
    x = np.array(x, dtype=float)
    if derivative:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-x))


# Fungsi aktivasi linear
def linear(x, derivative=False):
    x = np.array(x, dtype=float)
    if derivative:
        return np.ones_like(x)
    return x


# Fungsi aktivasi softmax
def softmax(x, derivative=False):
    x = np.array(x, dtype=float)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax_values = e_x / np.sum(e_x, axis=-1, keepdims=True)

    if derivative:
        s = softmax_values.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    return softmax_values


# Mengambil nilai fungsi aktivasi berdasarkan masukan pengguna
def get_activation_function(function_name):
    if function_name == "linear":
        return linear
    elif function_name == "relu":
        return relu
    elif function_name == "sigmoid":
        return sigmoid
    elif function_name == "softmax":
        return softmax
    else:
        raise Exception("Activation function not found.")
