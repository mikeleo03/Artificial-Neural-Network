import numpy as np
from .Layer import Dense
from graphviz import Digraph


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
            y_one_hot = np.mulsumeros((len(y), self.layers[-1].dimension))
            y_one_hot[np.arange(len(y)), y] = 1
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
            layer.weights += self.learning_rate * grad_w[i]
            layer.bias += self.learning_rate * grad_b[i]

    # Melakukan keseluruhan proses forward propagation
    def forward_propagation(self, X, y):
        X, y = np.array(X), np.array(y)

        # Melakukan one-hot encode kepada label
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
    def backward_propagation(self, X, y, y_prob):
        y = np.array(y)

        # Melakukan one-hot encode kepada label
        y_true = self.one_hot_encode(y)

        # Inisiasi nilai gradien bobot dan bias
        grad_w, grad_b = [], []

        # Melakukan kalkulasi gradien error terhadap weight di output layer
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

        dE_dw = np.dot(dE_dOut * dOut_dnet, dnet_dw)
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
            dE_dOut = np.dot(dE_dOut * dOut_dnet, dnetk_dOut)

            # ∂Out/∂net bergantung pada nilai turunan setiap fungsi aktivasi
            dOut_dnet = self.layers[layer_idx].activation(
                self.layers[layer_idx].mulsum, derivative=True
            )
            # ∂net/∂w, Jika dia adalah hidden layer pertama, maka nilainya adalah input
            if layer_idx == 0:
                dnet_dw = X
            else:  # Jika tidak maka nilai hasil propagation dari layer sebelumnya
                dnet_dw = self.layers[layer_idx - 1].output

            dE_dw = np.dot(dE_dOut * dOut_dnet, dnet_dw.T)
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
        verbose=True,
    ):
        self.inputs = np.array(X)
        self.learning_rate = learning_rate
        X, y = np.array(X), np.array(y)

        # Check if batch_size is valid
        if batch_size <= 0 or batch_size > len(X):
            raise Exception("Batch size invalid.")

        # Check if epochs is valid
        if epochs <= 0:
            raise Exception("Epochs value invalid.")

        # Check if error threshold is valid
        if error_threshold <= 0:
            raise Exception("Error threshold value invalid.")

        # Set the seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # Menghitung total loss
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            print(f"Epoch {epoch+1}/{epochs}")

            for i in range(0, len(X), batch_size):
                # Get batch
                size = min(batch_size, len(X) - i)

                # Forward propagation
                y_prob, _, loss = self.forward_propagation(
                    X[i : i + size], y[i : i + size]
                )

                # Backward propagation
                self.backward_propagation(X[i : i + size], y[i : i + size], y_prob)

                # Update epoch loss and metric
                epoch_loss += loss

                # Print progress bar
                progress = int(20 * (i + size) / len(X))
                progress_bar = "[" + "=" * progress + ">" + "-" * (29 - progress) + "]"
                if verbose:
                    print(
                        f"{i+size}/{len(X)} {progress_bar} - loss: {loss:.4f} - {self.metric}: {metric:.4f}",
                        end="\r",
                    )

            if verbose:
                print(
                    f"{len(X)}/{len(X)} [==============================] - loss: {epoch_loss:.4f} - {self.metric}: {epoch_metric:.4f}"
                )

            # Check apakah udah melewati nilai erro threshold
            if epoch_loss < error_threshold:
                print("[Stop] Error threshold is reached.")
                break

        # Jika berhneti, maka nilai maksimum iterasi telah tercapai
        print("[Stop] Maximum number of iteration reached.")

    # Mendapatkan rangkuman dari model yang terbentuk
    def summary(self):
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

            # Print the layer type
            dense_name = ""
            if counter == 0:
                dense_name = " dense (Dense)"
            else:
                dense_name = " dense_{counter} (Dense)"
            print(dense_name, end="")

            for i in range(len(" Layer (type)       ") - len(dense_name)):
                print(" ", end="")

            # Print the output shape
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
        print("===============================================")
        print(f"Total params: {total_params}")

    # Memberikan visualisasi hasil neural network.
    # Diharuskan untuk menginstall Grpahviz terlebih dahulu
    def visualize(self):
        if not self.built:
            raise ValueError("Model is not built yet")

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

    # Menyimpan model
    def save(self, name: str):
        model_file = f"model/{name}"
        with open(model_file, "wb") as f:
            pickle.dump(self, f)
