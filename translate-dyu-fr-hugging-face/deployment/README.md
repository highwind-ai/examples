# Deployment

This folder contains the resources required for deploying the trained model onto Highwind.

## Usage

> All commands below are run from this directory.

### Building your model image

This step builds the Kserve predictor image that contains your model.

1. First, make sure you have the trained model and tokenizer locally available. To get these, you can do one of the following:
    - Download the files from the [Hugging Face model repo](https://huggingface.co/MelioAI/dyu-fr-t5-small) and save them to `saved_model` (in the root directory of this example)
    - Run the inference notebooks located in the `notebooks` directory.

1. Copy over the trained model and its definition code into this folder so that it can be baked into the Docker container for serving

    ```bash
    cp -r ../saved_model .
    ```

1. Then build the container locally and give it a tag

    ```bash
    docker build -t local/highwind-examples/dyu-fr-inference:latest .
    ```

### Local testing

1. After building the Kserve predictor image that contains your model, spin it up to test your model inference

    ```bash
    docker compose up -d
    docker compose logs
    ```

1. Finally, send a payload to your model to test its response. To do this, use the `curl` cmmand to send a `POST` request with an example JSON payload.

    > Run this from another terminal (remember to navigate to this folder first)

    Linux/Mac Bash/zsh

    ```bash
    curl -X POST http://localhost:8080/v2/models/model/infer -H 'Content-Type: application/json' -d @./input.json
    ```

    Windows PowerShell

    ```PowerShell
    $json = Get-Content -Raw -Path ./input.json
    $response = Invoke-WebRequest -Uri http://localhost:8080/v2/models/model/infer -Method Post -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($json))
    $responseObject = $response.Content | ConvertFrom-Json
    $responseObject | ConvertTo-Json -Depth 10
    ```
