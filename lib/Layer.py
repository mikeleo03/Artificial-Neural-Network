import numpy as np
from .activation import get_activation_function

class Layer:
    '''
    Mengilustrasikan layer sebagai masukan untuk kelas Dense

    Atribut:
        dimention: Jumlah neuron dalam layer ini.
        built: Apakah layer sudah terbentuk atau belum.
        input_weight: Bobot dari setiap neuron dalam layer ini.
    '''
    
    # Inisiasi kelas
    def __init__(self, dimention):
        self.built = False
        self.dimention = dimention

    # Membangun layer dengan meng-assign nilai bobot
    def build(self, input_weight: np.array):
        self.input_weight = input_weight
        self.built = True

    # Melakukan forward propagation pada layer
    def call(self, inputs, *args, **kwargs):
        return inputs
    
    # Mengambil jumlah neuron dalam layer
    def get_dimention(self):
        return self.dimention
    
    # TODO : melakukan randomisasi weight yang normal, sesuai guideline di slide
    def random_weight(self, shape):
        return None

class Dense(Layer):
    '''
    Layer yang terhubung satu dengan yang lain

    Atribut:
        dimention: Jumlah neuron dalam layer ini.
        activation: Fungsi aktivasi yang digunakan.
        weights: Kumpulan bobot dari layer ini.
        bias: Bias dari layer ini.
        inputs: Input data yang digunakan untuk layer ini (dari json).
        input_shape: Bentuk input shape yang digunakan.
    '''
    
    # Inisiasi kelas
    def __init__(self, dimention, activation=None, input_shape=None):
        super().__init__(dimention)
        self.dimention = dimention
        self.activation = get_activation_function(activation)
        self.input_shape = input_shape
    
    # Membangun layer dengan assign nilai bobot dan bias
    # TODO : ubah, kalo input_weight nya gaada dia menginisasi random weight yang normal
    # HINT : utilisasi random_weight di kelas layer
    def build(self, input_weight: np.array, input_shape):
        self.weights = np.array(input_weight[1:])
        self.bias = np.array(input_weight[0])
        super().build(input_weight)
    
    # Melakukan feed-forward pada layer ini
    def call(self, inputs: np.array):
        self.inputs = inputs
        mulsum = np.dot(inputs, self.weights)
        mulsum += self.bias
        self.mulsum = mulsum
        self.output = self.activation(mulsum)
        return self.output
