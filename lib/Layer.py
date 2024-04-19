import numpy as np
import random
import math
from .activation import get_activation_function

class Layer:
    '''
    Mengilustrasikan layer sebagai masukan untuk kelas Dense

    Atribut:
        dimension: Jumlah neuron dalam layer ini.
        built: Apakah layer sudah terbentuk atau belum.
        input_weight: Bobot dari setiap neuron dalam layer ini.
    '''
    
    # Inisiasi kelas
    def __init__(self, dimension):
        self.built = False
        self.dimension = dimension

    # Membangun layer dengan meng-assign nilai bobot
    def build(self, input_weight: np.array):
        self.input_weight = input_weight
        self.built = True

    # Melakukan forward propagation pada layer
    def call(self, inputs, *args, **kwargs):
        return inputs
    
    # Mengambil jumlah neuron dalam layer
    def get_dimension(self):
        return self.dimension
    
    # Melakukan randomisasi weight yang normal, sesuai guideline di slide
    def random_weight(self, shape):
        weights = []
        for i in range(shape[0] + 1) :
            weights.append([])
            for j in range(shape[1]) :
                weights[i].append(random.uniform(-math.sqrt(6)/math.sqrt(shape[0] + shape[1]), math.sqrt(6)/math.sqrt(shape[0] + shape[1])))
        return weights

class Dense(Layer):
    '''
    Layer yang terhubung satu dengan yang lain

    Atribut:
        dimension: Jumlah neuron dalam layer ini.
        activation: Fungsi aktivasi yang digunakan.
        weights: Kumpulan bobot dari layer ini.
        bias: Bias dari layer ini.
        inputs: Input data yang digunakan untuk layer ini (dari json).
        input_shape: Bentuk input shape yang digunakan.
    '''
    
    # Inisiasi kelas
    def __init__(self, dimension, activation=None, input_shape=None):
        super().__init__(dimension)
        self.dimension = dimension
        self.activation = get_activation_function(activation)
        self.input_shape = input_shape
    
    # Bentuk representase Dense Layer, untuk keperluan print
    def __repr__(self):
        return ''.join([
            'Layer\n',
            f'activation = {self.activation.__name__},\n',
            f'weights =\n'
            f'{self.weights},\n',
            f'bias = {self.bias}'
        ])
    
    # Membangun layer dengan assign nilai bobot dan bias
    def build(self, input_weight: np.array = None):
        if input_weight is None :
            weights_arr = np.array(self.random_weight(shape=(self.input_shape[0], self.dimension)))
        else :
            weights_arr = np.array(input_weight)
        self.weights = weights_arr[1:]
        self.bias = weights_arr[0]
        super().build(weights_arr)
    
    # Melakukan feed-forward pada layer ini
    def call(self, inputs: np.array):
        self.inputs = inputs
        mulsum = np.dot(inputs, self.weights)
        mulsum += self.bias
        self.mulsum = mulsum
        self.output = self.activation(mulsum)
        return self.output
