class Sequential:
    """
    Model sequential seperti yang ada pada salindia kuliah

    Atribut:
        layers: Daftar layer yang ada pada model ini.
        result: Hasil akhir dari feed-forward.
        learning_rate: Learning rate dari model
    """

    # Inisiasi kelas
    def __init__(self, layers: list[Dense] = None) -> None:
        self.layers = layers
        if layers is None:
            self.layers: list[Dense] = []

    # Menambahkan layer ke dalam model
    def add(self, layer) -> None:
        self.layers.append(layer)

    # Melakukan feed-forward pada setiap layer dalam model
    def call(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.call(result)

        self.result = result
        return result

    # Melakukan one-hot encoding terhadap label y (untuk multiclass)
    def one_hot_encode(self, y):
        if len(y.shape) == 1:
            y_one_hot = np.array(
                [
                    [1 if y[i] == j else 0 for j in range(self.layers[-1].dimension)]
                    for i in range(len(y))
                ]
            )
            return y_one_hot
        else:
            y_one_hot = np.zeros((len(y), self.layers[-1].dimension))
            y_one_hot[np.arange(len(y)), y.astype(int)] = 1
            return y_one_hot

    # Melakukan kalkulasi loss berdasarkan fungsi aktivasi layer
    def compute_loss(self, y_true, y_prob):
        if self.layers[-1].activation.__name__ == "softmax":
            loss = (
                -np.sum(
                    np.log(
                        y_prob[np.arange(len(y_true)), y_true.argmax(axis=1)] + 1e-12
                    )
                )
                / y_true.shape[0]
            )
        else:
            loss = np.mean(0.5 * (y_true - y_prob) ** 2)
        return loss

    # Melakukan update weight dan bias pada backpropagation
    def update_weights(self, grad_w, grad_b):
        for i, layer in enumerate(self.layers):
            layer.weights -= self.learning_rate * grad_w[i]
            layer.bias -= self.learning_rate * grad_b[i]

    # Melakukan keseluruhan proses forward propagation
    def forward_propagation(self, X, y, encode):
        X, y = np.array(X), np.array(y)

        # Melakukan one-hot encoding hanya jika case dataset
        y_true = y
        if (encode):
            y_true = self.one_hot_encode(y)

        # Melakukan forward propagation
        y_prob = self.call(X)

        # Klasifikasi biner atau multiclass
        if y_prob.shape[-1] == 1:
            y_pred = np.array([0 if y_prob[i] > 0.5 else 1 for i in range(len(y_prob))])
        else:
            y_pred = np.argmax(y_prob, axis=-1)

        # Melakukan komputasi nilai loss
        loss = self.compute_loss(y_true, y_prob)

        return y_prob, y_pred, loss

    # Melakukan keseluruhan proses backward propagation
    def backward_propagation(self, X, y, y_prob, encode):
        y = np.array(y)

        # Melakukan one-hot encoding hanya jika case dataset
        y_true = y
        if (encode):
            y_true = self.one_hot_encode(y)

        # Inisiasi nilai gradien bobot dan bias
        grad_w, grad_b = [], []
        dOut_dnet = None
        dE_dOut = None

        # Melakukan kalkulasi gradien error terhadap weight di output layer
        if self.layers[-1].activation.__name__ == "softmax":
            # ∂E/∂w = ∂E/∂net * ∂net/∂w
            # ∂E/∂net khusus softmax menggunakan y_true
            dE_dnet1 = self.layers[-1].activation(self.layers[-1].mulsum, derivative=True, y_true=y_true)
            # ∂net/∂w, jika hanya ada satu layer (hanya layer output), maka nilainya adalah input
            if len(self.layers) == 1:
                dnet_dw = X
            else:  # Jika tidak maka nilai hasil propagation dari layer sebelumnya
                dnet_dw = self.layers[-2].output

            dE_dw = np.dot(dnet_dw.T, dE_dnet1)
            grad_w.append(dE_dw)

            # Melakukan kalkulasi gradien error terhadap bias di output layer
            dE_dnet = np.sum(dE_dnet1, axis=0)
            grad_b.append(dE_dnet)
        else:
            # ∂E/∂w = ∂E/∂Out * ∂Out/∂net * ∂net/∂w
            # ∂E/∂Out = -(tj - oj)
            dE_dOut = -1 * (y_true - y_prob)
            # ∂Out/∂net bergantung pada nilai turunan setiap fungsi aktivasi
            dOut_dnet = self.layers[-1].activation(self.layers[-1].mulsum, derivative=True)
            # ∂net/∂w, jika hanya ada satu layer (hanya layer output), maka nilainya adalah input
            if len(self.layers) == 1:
                dnet_dw = X
            else:  # Jika tidak maka nilai hasil propagation dari layer sebelumnya
                dnet_dw = self.layers[-2].output

            dE_dw = np.dot(dnet_dw.T, dE_dOut * dOut_dnet)
            grad_w.append(dE_dw)

            # Melakukan kalkulasi gradien error terhadap bias di output layer
            dE_dnet = np.sum(dE_dOut * dOut_dnet, axis=0)
            grad_b.append(dE_dnet)

        # Pemrosesan untuk setiap hidden layer
        # ∂E/∂w_ji = ∂E/∂net_j * ∂net_j/∂w
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            # ∂E/∂net_j = ∂E/∂net_k * ∂net_k/∂Out * ∂Out/∂net_j
            # ∂net_k/∂Out = wkj, weight dari layer sebelumnya
            dnetk_dOut = self.layers[layer_idx + 1].weights
            # Propagate the gradient backwards by multiplying with the gradient of the activation function
            if self.layers[layer_idx + 1].activation.__name__ == "softmax":
                dE_dOut = np.dot(dE_dnet, dnetk_dOut.T)
            else:
                dE_dOut = np.dot(dE_dOut * dOut_dnet, dnetk_dOut.T)

            # ∂Out/∂net bergantung pada nilai turunan setiap fungsi aktivasi
            dOut_dnet = self.layers[layer_idx].activation(
                self.layers[layer_idx].mulsum, derivative=True
            )
            # ∂net/∂w, Jika dia adalah hidden layer pertama, maka nilainya adalah input
            if layer_idx == 0:
                dnet_dw = X
            else:  # Jika tidak maka nilai hasil propagation dari layer sebelumnya
                dnet_dw = self.layers[layer_idx - 1].output
                
            dE_dw = np.dot(dnet_dw.T, dE_dOut * dOut_dnet)
            grad_w.insert(0, dE_dw)

            # Melakukan kalkulasi gradien error terhadap bias dari layer sebelumnya
            dE_dnet = np.sum(dOut_dnet * dE_dOut, axis=0)
            grad_b.insert(0, dE_dnet)

        # Update nilai bobot dan bias
        self.update_weights(grad_w, grad_b)

    # Menerima masukan dan model siap digunakan
    def fit(
        self,
        X,
        y,
        epochs: int = 100,
        batch_size: int = 10,
        learning_rate: float = 0.1,
        error_threshold: float = 0.1,
        random_state: int = 42,
        encode: bool = True,
        verbose: bool = True,
    ):
        self.inputs = np.array(X)
        self.learning_rate = learning_rate
        X, y = np.array(X), np.array(y)

        # Melakukan pengecekan apakah ukuran batch valid
        if batch_size <= 0 or batch_size > len(X):
            raise Exception("Batch size invalid.")

        # Melakukan pengecekan apakah masukan epoch valid
        if epochs <= 0:
            raise Exception("Epochs value invalid.")

        # Melakukan pengecekan apakah error threshold valid
        if error_threshold < 0:
            raise Exception("Error threshold value invalid.")

        # Mengatur seed awal proses randomisasi
        if random_state is not None:
            np.random.seed(random_state)

        # Menghitung total loss
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            print(f"Epoch {epoch+1}/{epochs}")
            
            # Memulai kalkulasi waktu
            time_start = time.time()

            for i in range(0, len(X), batch_size):
                # Mengambil batch
                size = min(batch_size, len(X) - i)

                # Melakukan forward propagation
                y_prob, _, loss = self.forward_propagation(X[i : i + size], y[i : i + size], encode)

                # Melakukan backward propagation
                self.backward_propagation(X[i : i + size], y[i : i + size], y_prob, encode)

                # Mengupdate nilai loss sebuah epoch
                epoch_loss += loss

                # Untuk keperluan mencetak progress bar
                progress = int(20 * (i + size) / len(X))
                bar = "[" + '\033[92m' + "━" * progress + ">" + "-" * (29 - progress) + '\033[0m' + "]"
                if verbose:
                    print(
                        f"{i+size}/{len(X)} {bar} - loss: {loss:.4f}",
                        end="\r",
                    )
                    
            # Waktu eksekusi selesai
            time_finish = time.time()
            
            # Apakah progress ingin ditampilkan?
            if verbose:
                print(
                    f"{len(X)}/{len(X)} [" + '\033[92m' + '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' + '\033[0m' + f"] - loss: {epoch_loss:.4f} - time: {time_finish - time_start:.4f} s"
                )                       

            # Cek apakah udah melewati nilai error threshold
            if epoch_loss / len(X) <= error_threshold:
                return "[Stop] Error threshold is reached."
                break

        # Jika berhenti, maka nilai maksimum iterasi telah tercapai
        return "[Stop] Maximum number of iteration reached."
    
    # Mendapatkan informasi seluruh layer
    def get_layers_info(self):
        for layer in self.layers:
            print(layer)
    
    # Melakukan prediksi kelas target
    def predict(self, X):
        X = np.array(X)

        # Melakukanorward propagation
        y_prob = self.call(X)

        if y_prob.shape[-1] == 1:
            y_pred = np.array([0 if y_prob[i] > 0.5 else 1 for i in range(len(y_prob))])
        else:
            y_pred = np.argmax(y_prob, axis=-1)

        return y_pred

    # Mendapatkan rangkuman dari model yang terbentuk
    def summary(self, full=False):
        print(' Model: "sequential"')
        for i in range(len(" Layer (type)        Output Shape       Param #")):
            print("-", end="")
        print()
        print(" Layer (type)        Output Shape       Param #")
        for i in range(len(" Layer (type)        Output Shape       Param #")):
            print("=", end="")
        print()

        total_params = 0
        counter = 0
        last_layer_dimension = 0
        for layer in self.layers:
            param_count = 0
            layer_dimension = layer.get_dimension()
            if counter == 0:
                param_count = (len(self.inputs[0]) + 1) * layer_dimension
            else:
                param_count = (last_layer_dimension + 1) * layer_dimension

            last_layer_dimension = layer_dimension

            # Mencetak tipe layer
            dense_name = ""
            if counter == 0:
                dense_name = " dense (Dense)"
            else:
                dense_name = f" dense_{counter} (Dense)"
            print(dense_name, end="")

            for i in range(len(" Layer (type)       ") - len(dense_name)):
                print(" ", end="")

            # Mencetak bentuk luaran
            print(f" (None, {layer_dimension})", end="")

            for i in range(
                len("Output Shape       ") - len(f" (None, {layer_dimension})")
            ):
                print(" ", end="")

            print(f" {param_count}", end="")

            for i in range(len("Param #") - len(str(param_count))):
                print(" ", end="")
            print()

            total_params += param_count
            counter += 1
        print("===============================================")
        print(f"Total params: {total_params}")
        
        if (full):
            print("\n-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-")
            print("Layer Summary")
            self.get_layers_info()

    # Memberikan visualisasi hasil neural network.
    # Diharuskan untuk menginstall Grpahviz terlebih dahulu
    def visualize(self):
        dot = Digraph(comment="FFNN")
        dot.attr(rankdir="LR", nodesep="1", ranksep="")
        dot.attr("node", shape="circle", width="0.4", height="0.4")

        # Jika hanya terdapat 1 layer (tidak ada hidden layer)
        if len(self.layers) == 1:
            # Semua Layer
            for i in range(len(self.inputs[0])):
                for j in range(len(self.result[0])):
                    weight = self.layers[0].weights[i][j]
                    dot.edge(
                        f"input{i}",
                        f"output{j}",
                        xlabel=f"{weight:.2f}",
                        color="#2ecc71",
                        xlabelfloat="true",
                    )

            # Bias
            for j in range(len(self.result[0])):
                weight = self.layers[0].bias[j]
                dot.edge(
                    f"bias0",
                    f"output{j}",
                    xlabel=f"{weight:.2f}",
                    color="#808080",
                    xlabelfloat="true",
                )

        # Jika tidak, artinya terdapat hidden layer
        else:
            # Input layer
            for i in range(len(self.inputs[0])):
                dot.node(f"input{i}", f"input{i}", color="#2ecc71")

            # Bias
            dot.node(f"bias0", f"bias0", color="#808080")

            # Hidden Layers
            for i in range(len(self.layers) - 1):
                for j in range(self.layers[i].dimension):
                    dot.node(f"hidden{i}{j}", f"hidden{i}{j}", color="#e67e22")

                if i == 0:
                    # Layer
                    for j in range(len(self.inputs[0])):
                        for k in range(self.layers[i].dimension):
                            weight = self.layers[i].weights[j][k]
                            dot.edge(
                                f"input{j}",
                                f"hidden{i}{k}",
                                xlabel=f"{weight:.2f}",
                                color="#2ecc71",
                            )

                    # Bias
                    for k in range(self.layers[i].dimension):
                        weight = self.layers[i].bias[k]
                        dot.edge(
                            f"bias{i}",
                            f"hidden{i}{k}",
                            xlabel=f"{weight:.2f}",
                            color="#808080",
                        )

                else:
                    # Layer
                    for j in range(self.layers[i - 1].dimension):
                        for k in range(self.layers[i].dimension):
                            weight = self.layers[i].weights[j][k]
                            dot.edge(
                                f"hidden{i-1}{j}",
                                f"hidden{i}{k}",
                                xlabel=f"{weight:.2f}",
                                color="#2ecc71",
                            )

                    # Bias
                    for k in range(self.layers[i].dimension):
                        weight = self.layers[i].bias[k]
                        dot.edge(
                            f"bias{i}",
                            f"hidden{i}{k}",
                            xlabel=f"{weight:.2f}",
                            color="#808080",
                        )

            # Output layer
            for i in range(len(self.result[0])):
                dot.node(f"output{i}", f"output{i}", color="#f1c40f")

            # Layer
            for i in range(self.layers[-2].dimension):
                for j in range(len(self.result[0])):
                    weight = self.layers[-1].weights[i][j]
                    dot.edge(
                        f"hidden{len(self.layers)-2}{i}",
                        f"output{j}",
                        xlabel=f"{weight:.2f}",
                        color="#f1c40f",
                    )

            # Bias
            for k in range(len(self.result[0])):
                weight = self.layers[-1].bias[k]
                dot.edge(
                    f"bias{len(self.layers)-1}",
                    f"output{k}",
                    xlabel=f"{weight:.2f}",
                    color="#808080",
                )

        # Simpan graph hasil visualisasi dalam png
        dot.render("output/ffnn_graph", format="png", cleanup=True)
        
    # Melakukan evaluasi hasil terhadap label
    def evaluate(self, expect):
        # Menguabh weights menjadi numpy
        labels = [np.array(layer) for layer in expect["final_weights"]]
        max_sse = 0.0000001

        converted_layers = []

        for layer in self.layers:
            # Membuat list converted
            new_layer = [layer.bias]
            new_layer.extend(layer.weights)
            # Dan menambahkannya ke converted_layers
            converted_layers.append(np.array(new_layer))

        print("Model evaluation (SSE)")
        print("=============================")

        # Pengecekan per layer
        for k in range(len(labels)):
            if converted_layers[k].shape != labels[k].shape:
                raise Exception('The size of the results is not equal to labels.')

            # Calculate SSE
            squared_errors = (labels[k] - converted_layers[k]) ** 2
            sse = np.sum(squared_errors)

            if len(labels) > 1:
                print(f"Layer {k + 1} --------------")
            print("sse        :", sse)
            print("sse status :", "Valid" if sse < max_sse else "Invalid")       

    # Menyimpan model
    def save(self, name: str):
        model_file = f"model/{name}"
        with open(model_file, "wb") as f:
            pickle.dump(self, f)