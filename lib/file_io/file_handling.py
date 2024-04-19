class JsonReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = None

    def read(self):
        with open(self.filename, 'r') as f:
            self.data = json.load(f)
        return self.data

    def get(self, index: int|str):
        return self.data[index]

    def length(self):
        return len(self.data)
    
class JsonParser:
    def parse_model_config(self, json_path: str) -> ModelConfig:
        model_config: ModelConfig = JsonReader(json_path).read()
        return model_config
    
class Layers(TypedDict):
    number_of_neurons: int
    activation_function : str

class ModelData(TypedDict):
    input_size: int
    layers: Layers

class LearningParameters(TypedDict): 
    learning_rate: float
    batch_size: int
    max_iteration: int
    error_threshold: float

class Case(TypedDict):
    model: ModelData
    input: list[list[list[float]]]
    initial_weights: list[list[list[float]]]
    target: list[list[list[float]]]
    learning_parameters: LearningParameters
class Expect(TypedDict):
    stopped_by: str
    final_weights: list[list[list[float]]]
    
class ModelConfig(TypedDict):
    case: Case
    expect: Expect