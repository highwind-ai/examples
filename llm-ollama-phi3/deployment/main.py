"""Kserve inference script."""

import argparse

from kserve import (
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    model_server,
)
from kserve.utils.utils import generate_uuid
from langchain_community.chat_models import ChatOllama

LLM_NAME = "phi3:3.8b"


class MyModel(Model):
    """Kserve inference implementation of model."""

    def __init__(self, name: str):
        """Initialise model."""
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        """Reconstitute model from disk."""
        self.model = ChatOllama(model=LLM_NAME)
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> str:
        """Preprocess inference request."""
        input_text = payload.inputs[0].data[0]
        return input_text

    def predict(self, data: str, *args, **kwargs) -> InferResponse:
        """Pass inference request to model to make prediction."""
        # Model prediction on scaled features
        result = self.model.invoke(data)
        response_id = generate_uuid()
        infer_output = InferOutput(
            name="output-0", shape=[1], datatype="STR", data=result
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name", default="model", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    ModelServer().start([model])
