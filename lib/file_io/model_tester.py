class model_tester:
    @staticmethod
    def test(test_case: str):
        model_config: ModelConfig = JsonParser().parse_model_config(test_case)
        model_factory = ModelFactory()
        model = model_factory.build(model_config)

        case = model_config['case']
        learning_parameters = case["learning_parameters"]
        data = case['input']
        target = case['target']

        stop_reason = model.fit(
            data, 
            target, 
            epochs=learning_parameters['max_iteration'], 
            batch_size=learning_parameters['batch_size'], 
            learning_rate=learning_parameters['learning_rate'], 
            error_threshold=learning_parameters['error_threshold'], 
            random_state=42, 
            verbose=True
        )
        
        print("===================================================")
        model.summary()
        print("Expected final weights:",
            model_config['expect'].get("final_weights"))

        print("===================================================")
        print(stop_reason)

        print("Expected stop reason: ", model_config['expect']['stopped_by'])
        model.draw()