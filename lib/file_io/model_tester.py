class ModelTester:
    """
    Melakukan pengujain berdasarkan model yang menjadi masukan kasus uji asisten.
    """
    
    # Metode statik untuk melakukan pengujian model
    @staticmethod
    def test(test_case: str):
        # Membaca konfigurasi model dan menginisasi model factory
        model_config: ModelConfig = JSONReader(test_case).read()
        model_factory = ModelFactory()
        model = model_factory.build(model_config)

        # Mengambil komponen config yang diperlukan
        case = model_config['case']
        learning_parameters = case["learning_parameters"]
        data = case['input']
        target = case['target']

        # Melakukan model fiitting model dan mendapatkan alasan berhenti
        stop_reason = model.fit(
            data, 
            target, 
            epochs=learning_parameters['max_iteration'], 
            batch_size=learning_parameters['batch_size'], 
            learning_rate=learning_parameters['learning_rate'], 
            error_threshold=learning_parameters['error_threshold'], 
            random_state=42, 
            encode=False,
            verbose=True
        )
        
        # Logging Hasil
        print("\n-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-")
        print("Model Summary")
        model.summary(full=True)
        print("Expected final weights:",
            model_config['expect'].get("final_weights"))

        print("\n-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-=x=-")
        print(stop_reason)

        print("Expected stop reason: ", model_config['expect']['stopped_by'])