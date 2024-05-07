import os
import re
import argparse
from typing import Dict, List

import torch
from joeynmt.prediction import predict, prepare
from joeynmt.config import load_config, parse_global_args
from kserve.utils.utils import generate_uuid
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse


MODEL_CONFIG_PATH = "/app/saved_model/config.yaml"
CHARS_TO_REMOVE_REGEX = '[!"&\(\),-./:;=?+.\n\[\]]'


def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
    return text.strip()


class JoeyNMTModelDyuFr:
    """
    JoeyNMTModelDyuFr which load JoeyNMT model for inference.

    :param config_path: Path to YAML config file
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    """
    def __init__(self, config_path: str, n_best: int = 1) -> None:
        seed = 42
        torch.manual_seed(seed)
        cfg = load_config(config_path)
        args = parse_global_args(cfg, rank=0, mode="translate")
        self.args = args._replace(test=args.test._replace(n_best=n_best))
        # build model
        self.model, _, _, self.test_data = prepare(self.args, rank=0, mode="translate")

    def _translate_data(self) -> List[str]:
        _, _, hypotheses, _, _, _ = predict(
            model=self.model,
            data=self.test_data,
            compute_loss=False,
            device=self.args.device,
            rank=0,
            n_gpu=self.args.n_gpu,
            normalization="none",
            num_workers=self.args.num_workers,
            args=self.args.test,
            autocast=self.args.autocast,
        )
        return hypotheses

    def translate(self, sentence) -> List[str]:
        """
        Translate the given sentence.

        :param sentence: Sentence to be translated
        :return:
        - translations: (list of str) possible translations of the sentence.
        """
        self.test_data.set_item(sentence.strip())
        translations = self._translate_data()
        assert len(translations) == len(self.test_data) * self.args.test.n_best
        self.test_data.reset_cache()
        return translations


class MyModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        # Instantiate model
        self.model = JoeyNMTModelDyuFr(config_path=MODEL_CONFIG_PATH, n_best=1)
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> List[str]:
        # Get data from payload
        infer_inputs: List[str] = payload.inputs[0].data
        print(f"** infer_input ({type(infer_inputs)}): {infer_inputs}")

        cleaned_texts: List[str] = [clean_text(i) for i in infer_inputs]
        print(f"** cleaned_text ({type(cleaned_texts)}): {cleaned_texts}")
        return cleaned_texts

    def predict(self, data: List[str], *args, **kwargs) -> InferResponse:
        response_id = generate_uuid()
        results: List[str] = [self.model.translate(sentence=s)[0] for s in data]
        print(f"** result ({type(results)}): {results}")

        infer_output = InferOutput(name="output-0", shape=[len(results)], datatype="STR", data=results)
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