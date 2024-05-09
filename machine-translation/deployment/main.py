#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, Union, List
from kserve import Model, ModelServer, model_server, InferInput, InferRequest, InferOutput, InferResponse
from kserve.utils.utils import generate_uuid

from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM


class MachineTranslation(Model):
    def __init__(self, name: str, protocol: str):
        super().__init__(name)
        self.model = None
        self.tokenizer = None
        self.protocol = protocol
        self.ready = False
        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tf_saved_model")
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_saved_model/")
        self.ready = True

    # def preprocess(self, payload: InferRequest, headers: Dict[str, str] = None) \
    #         -> Union[Dict, InferRequest]:
    #     input_tensors = [image_transform(instance) for instance in payload.inputs[0].data]
    #     print(f"Input tensors: \n{input_tensors}")
    #     infer_inputs = [InferInput(name="FEATURES", datatype='FP32', shape=[len(input_tensors), 10],
    #                                data=input_tensors)]
    #     infer_request = InferRequest(model_name=self.name, infer_inputs=infer_inputs)
    #     return infer_request

    def predict(self, payload: InferRequest, headers: Dict[str, str] = None) -> InferResponse:
        print("Starting predict..")
        print(f"{payload.inputs[0].data[0]}")

        response_id = generate_uuid()
        inputs = self.tokenizer(payload.inputs[0].data[0], return_tensors="tf").input_ids
        outputs = self.model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        infer_output = InferOutput(name="PREDICTION__0", datatype='BYTES', shape=[1], data=[translated_text])
        infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        return infer_response

    # def postprocess(self, infer_response: Union[Dict, ModelInferResponse], headers: Dict[str, str] = None) \
    #         -> Union[Dict, InferResponse]:
    #     if "request-type" in headers and headers["request-type"] == "v1":
    #         if self.protocol == PredictorProtocol.REST_V1.value:
    #             return infer_response
    #         else:
    #             res = super().postprocess(infer_response, headers)
    #             return {"predictions": res["outputs"][0]["data"]}
    #     else:
    #         return super().postprocess(infer_response, headers)


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--protocol", help="The protocol for the predictor", default="v2"
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MachineTranslation(args.model_name,
                             protocol=args.protocol)
    ModelServer().start([model])
