import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib.file_io.file_handling import *
from lib.Sequential import Sequential
from lib.Layer import Dense

if __name__ == "__main__":
    # Pastikan data berada pada folder "test"
    filename = "multilayer.json"
    data = load_json(filename)

    # Mengambil beberapa data yang diperlukan dari data uji
    layers = data["case"]["model"]
    weights = data["case"]["weights"]
    inputs = data["case"]["input"]
    data_label = data["expect"]

    # Membangun model
    model = Sequential()
    for i in range (len(layers['layers'])):
        layer = layers['layers'][i]
        weight = weights[i]
        dense_layer = Dense(layer["number_of_neurons"], activation=layer["activation_function"])
        dense_layer.build(weight)
        model.add(dense_layer)

    # Compile-fit model dengan data masukan
    model.fit(inputs)
    
    # Mendapatkan informasi model
    model.summary()
    
    # Melakukan forward propagation
    y_prob = model.forward_propagation()
    print(y_prob)
    
    # Melakukan visualisasi model struktur jarigan
    model.visualize()

    # Mengambil gambar hasil visualisasi model untuk ditampilkan
    image_path = "output/ffnn_graph.png"

    # Load and display the image
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
