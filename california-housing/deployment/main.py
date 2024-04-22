import argparse
from typing import Dict
import numpy as np
import joblib
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
from kserve.utils.utils import generate_uuid


class MyModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        # Load feature scaler and trained model
        self.scaler = joblib.load("/app/saved_model/scaler.joblib")
        self.model = joblib.load("/app/saved_model/model.joblib")
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> np.ndarray:
        # Scale input features
        infer_input = payload.inputs[0]
        raw_data = np.array(infer_input.data)
        scaled_data = self.scaler.transform(raw_data)
        return scaled_data

    def predict(self, data: np.ndarray, *args, **kwargs) -> InferResponse:
        # Model prediction on scaled features
        result = self.model.predict(data)
        result = result.astype(np.float32)
        response_id = generate_uuid()
        infer_output = InferOutput(name="output-0", shape=list(result.shape), datatype="FP32", data=result)
        infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        return infer_response

    # def postprocess(self, payload, *args, **kwargs):
    #     # Optionally postprocess payload
    #     return payload


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name",
    default="model",
    help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    ModelServer().start([model])