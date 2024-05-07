# Translate Dyula to French (Hugging Face)

An example of a machine translation model that translates Dyula to French using the [JoeyNMT framework](https://github.com/joeynmt/joeynmt).

This following example is based on [this Github repo](https://github.com/data354/koumakanMT-challenge) that was kindly created by [data354](https://data354.com/en/).

> Training times (6 epochs):
>
> - Google Colab: 26 min
>   - GPU: T4 (15 GB)
>   - RAM: 12.7 GB
>   - CPU: 1x Intel Xeon (x86_64)
> - Gitpod large: 67 min
>   - GPU: None
>   - RAM: 16 GB
>   - CPU: 8x AMD EPYC 7B13 (x86_64)

## Usage

### Running notebooks

> Make sure you run your notebooks in the relevant virtual environments created below.

1. Set up your environment with the required dependencies.

- For data processing and training the model, set up the `train` environment by running the following:

    ```shell
    # Use Python 3.11

    # Initial setup
    python -m venv .train.venv && source .train.venv/bin/activate
    pip install -r training/train-requirements.txt

    # Activate after setup (run every time)
    source .train.venv/bin/activate
    ```

- For model serving and inference, set up the `serve` environment by running the following:

    > Make sure you uncomment the `ipykernel` requirement in the `requirements.txt` file before running the commands below.

    ```shell
    # Use Python 3.11

    # Initial setup
    python -m venv .serve.venv && source .serve.venv/bin/activate
    pip install -r deployment/serve-requirements.txt

    # Activate after setup (run every time)
    source .serve.venv/bin/activate
