# Translate Dyula to French (Hugging Face)

An example of a machine translation model that translates English to French. This model is directly used from [this Hugging Face repo](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) without further training.

## Usage

### Running notebooks

> Make sure you run your notebooks in the relevant virtual environments created below.

1. Set up your environment with the required dependencies.

> You can run these commands from any directory but we recommend running them from the example root directory (where this README is).

- For model serving and inference, set up the `serve` environment by running the following:

    > Make sure you uncomment the `ipykernel` requirement in the `requirements.txt` file before running the commands below if you want to run the inference notebook.

    ```shell
    # Use Python 3.10

    # Initial setup
    python -m venv .serve.venv && source .serve.venv/bin/activate
    pip install -r deployment/serve-requirements.txt

    # Activate after setup (run every other time)
    source .serve.venv/bin/activate
    ```
