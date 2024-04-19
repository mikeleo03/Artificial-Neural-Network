class ModelFactory:
    def build(self, model_config: ModelConfig) -> Model:
        # Get the model case
        case = model_config["case"]
        layers = case["model"]
        weights = case["initial_weights"]
        inputs = case["input"]
        
        # Membangun model ANN
        model = Sequential()
        for i in range (len(layers['layers'])):
            layer = layers['layers'][i]
            weight = weights[i]
            dense_layer = Dense(layer["number_of_neurons"], activation=layer["activation_function"])
            dense_layer.build(weight)
            model.add(dense_layer)
        
        return model
    
    def load(self, path: str) -> Model:
        with open(path, 'rb') as f:
            return pickle.load(f)