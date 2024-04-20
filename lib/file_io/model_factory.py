class ModelFactory:
    """
    Membangun model berdasarkan model config yang tersedia dari kasus uji asisten
    """
    
    # Membangun model berdasarkan model config yang tersedia
    def build(self, model_config: ModelConfig) -> Sequential:
        # Mengambil elemen model config yang diperlukan
        case = model_config["case"]
        layers = case["model"]
        weights = case["initial_weights"]
        inputs = case["input"]
        
        # Membangun model ANN
        model = Sequential()
        for i in range (len(layers['layers'])):
            layer = layers['layers'][i]
            weight = weights[i]
            dense_layer = Dense(layer["number_of_neurons"], activation=layer["activation_function"], input_weight=weight)
            model.add(dense_layer)
        
        # Kembalikan model
        return model
            
    # Memuat ulang model yang disimpan dalam path file
    def load(self, name: str) -> Sequential:
        model_file = f"model/{name}"
        with open(model_file, 'rb') as f:
            return pickle.load(f)