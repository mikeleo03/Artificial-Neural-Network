import numpy as np
from graphviz import Digraph

class Sequential:
    '''
    Model sequential seperti yang ada pada salindia kuliah

    Atribut:
        layers: Daftar layer yang ada pada model ini.
        built: Apakah model sudah terbentuk atau belum.
        result: Hasil akhir dari feed-forward.
    '''
    
    # Inisiasi kelas
    def __init__(self):
        self.layers = []
        self.built = False
        self.result = None

    # Menambahkan layer ke dalam model
    def add(self, layer):
        self.layers.append(layer)
    
    # Melakukan feed-forward pada setiap layer dalam model
    def call(self, inputs):
        self.result = inputs
        for layer in self.layers:
            self.result = layer.call(self.result)
        
        return self.result
    
    # Menerima masukan dan model siap digunakan
    def fit(self, inputs):
        self.inputs = np.array(inputs)
        self.built = True
    
    # Mendapatkan rangkuman dari model yang terbentuk
    def summary(self):
        if not self.built:
            raise ValueError('Model is not built yet')

        print(' Model: "sequential"')
        for i in range(len(' Layer (type)        Output Shape       Param #')):
            print('-', end='')
        print()
        print(' Layer (type)        Output Shape       Param #')
        for i in range(len(' Layer (type)        Output Shape       Param #')):
            print('=', end='')
        print()
        
        total_params = 0
        counter = 0
        last_layer_dimention = 0
        for layer in self.layers:
            param_count = 0
            layer_dimention = layer.get_dimention()
            if (counter == 0):
                param_count = (len(self.inputs[0]) + 1) * layer_dimention
            else:
                param_count = (last_layer_dimention + 1) * layer_dimention
            
            last_layer_dimention = layer_dimention
            
            # Print the layer type
            dense_name = ""
            if (counter == 0):
                dense_name = " dense (Dense)"
            else:
                dense_name = " dense_{counter} (Dense)"
            print(dense_name, end='')

            for i in range(len(' Layer (type)       ') - len(dense_name)):
                print(' ', end='')

            # Print the output shape
            print(f' (None, {layer_dimention})', end='')
            
            for i in range(len('Output Shape       ') - len(f' (None, {layer_dimention})')):
                print(' ', end='')
            
            print(f' {param_count}', end='')

            for i in range(len('Param #') - len(str(param_count))):
                print(' ', end='')
            print()

            total_params += param_count
        print('===============================================')
        print(f'Total params: {total_params}')
    
    # Melakukan forward propagation berdasarkan input
    def forward_propagation(self):
        y_prob = self.call(self.inputs)

        return y_prob
    
    # Memberikan visualisasi hasil neural network. 
    # Diharuskan untuk menginstall Grpahviz terlebih dahulu
    def visualize(self):
        if not self.built:
            raise ValueError('Model is not built yet')
        
        dot = Digraph(comment='FFNN')
        dot.attr(rankdir='LR', nodesep='1', ranksep='')
        dot.attr('node', shape='circle', width='0.4', height='0.4')
        
        # Jika hanya terdapat 1 layer (tidak ada hidden layer)
        if len(self.layers) == 1:
            # Semua Layer
            for i in range(len(self.inputs[0])):
                for j in range(len(self.result[0])):
                    weight = self.layers[0].weights[i][j]
                    dot.edge(f'input{i}', f'output{j}', xlabel=f'{weight:.2f}', color='#2ecc71', xlabelfloat='true')

            # Bias
            for j in range(len(self.result[0])):
                weight = self.layers[0].bias[j]
                dot.edge(f'bias0', f'output{j}', xlabel=f'{weight:.2f}', color='#808080', xlabelfloat='true')
        
        # Jika tidak, artinya terdapat hidden layer 
        else :
            # Input layer
            for i in range(len(self.inputs[0])):
                dot.node(f'input{i}', f'input{i}', color='#2ecc71')
            
            # Bias
            dot.node(f'bias0', f'bias0', color='#808080')

            # Hidden Layers
            for i in range(len(self.layers) - 1):
                for j in range(self.layers[i].dimention):
                    dot.node(f'hidden{i}{j}', f'hidden{i}{j}', color='#e67e22')

                if i == 0:
                    # Layer
                    for j in range(len(self.inputs[0])):
                        for k in range(self.layers[i].dimention):
                            weight = self.layers[i].weights[j][k]
                            dot.edge(f'input{j}', f'hidden{i}{k}', xlabel=f'{weight:.2f}', color='#2ecc71')
                    
                    # Bias
                    for k in range(self.layers[i].dimention):
                        weight = self.layers[i].bias[k]
                        dot.edge(f'bias{i}', f'hidden{i}{k}', xlabel=f'{weight:.2f}', color='#808080')

                else:
                    # Layer
                    for j in range(self.layers[i-1].dimention):
                        for k in range(self.layers[i].dimention):
                            weight = self.layers[i].weights[j][k]
                            dot.edge(f'hidden{i-1}{j}', f'hidden{i}{k}', xlabel=f'{weight:.2f}', color='#2ecc71')
                    
                    # Bias
                    for k in range(self.layers[i].dimention):
                        weight = self.layers[i].bias[k]
                        dot.edge(f'bias{i}', f'hidden{i}{k}', xlabel=f'{weight:.2f}', color='#808080')

            # Output layer
            for i in range(len(self.result[0])):
                dot.node(f'output{i}', f'output{i}', color='#f1c40f')

            # Layer
            for i in range(self.layers[-2].dimention):
                for j in range(len(self.result[0])):
                    weight = self.layers[-1].weights[i][j]
                    dot.edge(f'hidden{len(self.layers)-2}{i}', f'output{j}', xlabel=f'{weight:.2f}', color='#f1c40f')
            
            # Bias
            for k in range(len(self.result[0])):
                weight = self.layers[-1].bias[k]
                dot.edge(f'bias{len(self.layers)-1}', f'output{k}', xlabel=f'{weight:.2f}', color='#808080')

        # Simpan graph hasil visualisasi dalam png
        dot.render("output/ffnn_graph", format="png", cleanup=True)
    
    # Melakukan evaluasi hasil terhadap label
    def evaluate(self, expect):
        # Ambil nilai expectednya
        labels = expect["output"]
        max_sse = expect["max_sse"]
        
        # Lakukan pengecekan apakah ukuran labelnya sama
        if len(self.result) != len(labels):
            raise Exception('The size of the results is not equal to labels.')
        
        print(" Model evaluation")
        print("=============================")
        
        # Handle the batch input
        for k in range (len(labels)):
            # Inisiasi variabel
            total = 0
            correct = 0
            
            # Validasi isi label
            if len(self.result[k]) != len(labels[k]):
                raise Exception('The size of the results is not equal to labels.')
            
            # Kalkulasi akurasi
            for j in range (len(self.result[k])):
                if abs(round(self.result[k][j], 2) - round(labels[k][j], 2)) <= 0.05:
                    correct += 1
                    
                total += 1
                
            accuracy = correct / total * 100
            
            # Kalkulasi SSE
            squared_errors = (labels[k] - self.result[k]) ** 2
            sse = np.sum(squared_errors)

            if (len(labels) > 1):
                print(f"Input {k + 1} --------------")
            print(f" accuracy   : {round(accuracy, 2)}%")
            print(" sse        :", sse)
            print(" sse status :", "Valid" if sse < max_sse else "Invalid")
        