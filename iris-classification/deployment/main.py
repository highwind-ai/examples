import argparse
from typing import Dict
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
from kserve.utils.utils import generate_uuid
import os
import pandas as pd
import joblib
import json
import numpy as np

ARTIFACT_SAVE_DIR = "./saved_model/"


class MyModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.scaler = None
        self.ready = False
        self.load()

    def load(self):
        # Load scaler and model
        self.scaler = joblib.load(os.path.join(ARTIFACT_SAVE_DIR, "scaler.joblib"))
        self.model = joblib.load(os.path.join(ARTIFACT_SAVE_DIR, "model.joblib"))
        self.ready = True

    def preprocess(self, payload: InferRequest, headers: Dict[str, str] = None) -> np.ndarray:
        # Get data from payload
        infer_input = payload.inputs[0]
        raw_data = np.array(infer_input.data)

        # Scale data
        scaled_data = self.scaler.transform(raw_data)

        print(f"** scaled_data ({type(scaled_data)}): {scaled_data}")
        return scaled_data

    def predict(self, data: np.ndarray, headers: Dict[str, str] = None) -> InferResponse:
        response_id = generate_uuid()
        result = self.model.predict(data)
        infer_output = InferOutput(name="output-0", shape=list(result.shape), datatype="INT64", data=result)
        infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        return infer_response

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