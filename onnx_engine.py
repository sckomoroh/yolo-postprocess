import onnxruntime as ort
import ast

class OnnxEngine:
    def __init__(self, model: str):
        self.class_names = {}
        self.sesstion = ort.InferenceSession(model)

    def dump_inputs(self):
        inputs = self.sesstion.get_inputs()
        for input in inputs:
            print(f'Name: {input.name} Shape: {input.shape}')

    def dump_outputs(self):
        inputs = self.sesstion.get_outputs()
        for input in inputs:
            print(f'Name: {input.name} Shape: {input.shape}')

    def infer(self, tensor: dict):
        return self.sesstion.run(None, tensor)

    def get_class_name(self, index: int):
        return self.class_names[index]

    def read_names(self, filename: str):
        with open(filename, "r") as f:
            self.class_names = ast.literal_eval(f.read())