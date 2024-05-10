"""Kserve inference script."""

import argparse
from kserve import (
    Model,
    ModelServer,
    model_server,
    InferRequest,
    InferOutput,
    InferResponse,
)
from kserve.utils.utils import generate_uuid
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM


class MachineTranslation(Model):
    """Kserve inference implementation of model."""

    def __init__(self, name: str, protocol: str):
        """Initialise model."""
        super().__init__(name)
        self.model = None
        self.tokenizer = None
        self.protocol = protocol
        self.ready = False
        self.load()

    def load(self):
        """Reconstitute model from disk."""
        self.tokenizer = AutoTokenizer.from_pretrained("tf_saved_model")
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_saved_model/")
        self.ready = True

    def predict(self, payload: InferRequest, *args, **kwargs) -> InferResponse:
        """Pass inference request to model to make prediction."""
        print("Starting predict..")
        print(f"{payload.inputs[0].data[0]}")

        response_id = generate_uuid()
        inputs = self.tokenizer(
            payload.inputs[0].data[0], return_tensors="tf"
        ).input_ids
        outputs = self.model.generate(
            inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
        )
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        infer_output = InferOutput(
            name="PREDICTION__0", datatype="BYTES", shape=[1], data=[translated_text]
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument("--protocol", help="The protocol for the predictor", default="v2")
parser.add_argument("--model_name", help="The name that the model is served under.")
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MachineTranslation(args.model_name, protocol=args.protocol)
    ModelServer().start([model])
