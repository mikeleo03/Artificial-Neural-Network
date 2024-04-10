import numpy as np

# Fungsi aktivasi reLU
def relu(x, derivative=False):
    if derivative:
        # TODO : implement
        return None
    
    return np.maximum(0, x)

# Fungsi aktivasi sigmoid
def sigmoid(x, derivative=False):
    if derivative:
        # TODO : implement
        return None
    
    return 1 / (1 + np.exp(-x))

# Fungsi aktivasi linear
def linear(x, derivative=False):
    if derivative:
        # TODO : implement
        return None
    
    return x

# Fungsi aktivasi softmax
def softmax(x, derivative=False):
    if derivative:
        # TODO : implement
        return None
    
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Mengambil nilai fungsi aktivasi berdasarkan masukan pengguna
def get_activation_function(function_name):
    if function_name == 'linear':
        return linear
    elif function_name == 'relu':
        return relu
    elif function_name == 'sigmoid':
        return sigmoid
    elif function_name == 'softmax':
        return softmax
    else:
        raise Exception('Activation function not found.')
