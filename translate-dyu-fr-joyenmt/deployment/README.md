# Deployment

This folder contains the resources required for deploying the trained model onto Highwind.

## Usage

> All commands below are run from this directory.

### Building your model image

This step builds the Kserve predictor image that contains your model.

1. First, copy over the trained model and its definition code into this folder so that it can be baked into the Docker container for serving

    > 💡 Note that when running the `01-train.ipynb` notebook, several model checkpoints, vocabulary files, tokeniser models, etc. are saved. These are not all necessary for baking into the serve container for inference. Only the necessary files from this set are copied to a folder called `lean_model`, which is used to construct the serving container. The essential files include the following:
    >
    > - Best model checkpoint (e.g. `510.ckpt` -> rename to `best.ckpt`)
    > - JoyNMT config file (`config.yaml`)
    > - Tokenizer (`sp.model`)
    > - Vocabulary file (`vocab.txt`)

    > 💡 Remember to change the paths in the config.yml file to point to paths in the serving container (e.g. `/app/saved_model`, check the Dockerfile).

    > 💡 Note when downloading the model from HuggingFace via the `02-inference.ipynb` notebook and you see the file called `best.ckpt` has a symbolic link then please proceed to run the following command, to replace the link with actual model, ensuring you're within the `translate-dyu-fr-joyenmt/saved_model/lean_model` directory.
    >
    > ```bash
    > mv best.ckpt _best.ckpt && mv $(readlink -f _best.ckpt) best.ckpt && rm _best.ckpt
    > ```

    ```bash
    cp -R ../saved_model/lean_model ./saved_model
    ```

1. Then build the container locally and give it a tag

    ```bash
    docker build -t local/highwind-examples/dyu-fr-inference .
    ```

### Local testing

1. After building the Kserve predictor image that contains your model, spin it up to test your model inference

    ```bash
    docker compose up -d
    docker compose logs
    ```

1. Finally, send a payload to your model to test its response. To do this, use the `curl` cmmand to send a `POST` request with an example JSON payload.

    >  Run this from another terminal (remember to navigate to this folder first)

    Linux/Mac Bash/zsh

    ```bash
    curl -X POST http://localhost:8080/v2/models/model/infer -H 'Content-Type: application/json' -d @./input.json
    ```

    Windows PowerShell

    ```PowerShell
    $json = Get-Content -Raw -Path ./input.json
    Invoke-WebRequest -Uri http://localhost:8080/v2/models/model/infer -Method Post -ContentType 'application/json' -Body $json
    ```
