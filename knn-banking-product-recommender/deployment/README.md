# Deployment

This folder contains the resources required for deploying the trained model onto Highwind.

## Predictor

> All commands below are run from this directory.

### Building your model image

This step builds the Kserve predictor image that contains your model.

1. First, make sure you have the knn indices and preprocessed files locally available. To get these, you can do one of the following:
    - Run the following notebooks located in the `notebooks` directory:
        - `data_exploration_santander_product_ds.ipynb`: This will run the entire notebook process and save the files into the `saved_model` folder.

        - `download_files.ipynb`: This will download the saved files from HuggingFace into the `saved_model` folder.

1. Copy over the saved model folder contents and its definition code into this folder so that it can be baked into the Docker container for serving.
   > **NB:** Ensure you are within the `deployment` folder before running the below command

    ```bash
    cp -r ../saved_model .
    ```

1. Then build the container locally and give it a tag

    ```bash
    docker build -t local/highwind-examples/knn-santander-inference .
    ```

### Local testing

1. After building the Kserve predictor image that contains your model, spin it up to test your model inference

    ```bash
    docker compose up -d
    docker compose logs
    ```

1. Finally, send a payload to your model to test its response. To do this, use the `curl` command to send a `POST` request with an example JSON payload.

    **NB:** Run this from another terminal (remember to navigate to this folder first)

    Linux/Mac Bash/zsh

    ```bash
    curl -X POST http://localhost:8080/v2/models/model/infer -H 'Content-Type: application/json' -d @./input.json
    ```

    - It may take a **few seconds** for the server to start. To verify that the server is running, you can execute the following command:

        ```bash
        curl http://localhost:8080/
        ```

    - If you see `{"status":"alive"}%`, the server is ready, and you can proceed to test the response.



    Windows PowerShell

    ```PowerShell
    $json = Get-Content -Raw -Path ./input.json
    $response = Invoke-WebRequest -Uri http://localhost:8080/v2/models/model/infer -Method Post -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($json))
    $responseObject = $response.Content | ConvertFrom-Json
    $responseObject | ConvertTo-Json -Depth 10
    ```

## Gradio App

The Gradio app is a simple graphical interface for testing the model. It is not required for deployment, but it can be added to the Highwind Use Case to allow for easier testing.

### Local testing

> All commands below are run from this directory.
> You can also test how the app works without running inference on the deployed Highwind Use Case by setting function of the function of the `submit_button.click` to `get_mock_product_recommendation` instead of `get_product_recommendation` (set this in `demo.py`).

1. Install the dependencies in a virtual environment

    > This section requires Python 3.12 or higher

    ```bash
    python -m venv .gradio.venv && source .gradio.venv/bin/activate
    pip install -r gradio.requirements.txt
    ```

1. Uncomment the last line of the `demo.py` file so that the app launches locally when following the next step.

    > ⚠️ Warning: Do not forget to comment the `demo.launch()` line of the `demo.py` before deploying to Highwind. This step is only for local testing!

1. Run the app (from the virtual environment)

    ```bash
    python demo.py
    ```

1. Open your browser and navigate to [`http://127.0.0.1:7860`](http://127.0.0.1:7860) to access the app.

### Deploying the app on Highwind

1. Create a Gradio Asset on Highwind
    1. On Highwind, create a new Asset and select "Gradio App" as the Asset type.
    2. Upload the `demo.py` to this asset.
        > ⚠️ Warning: Remember to comment the `demo.launch()` line of the `demo.py` before uploading to Highwind.

1. Create a new Highwind Use Case
    1. On Highwind, create a new Use Case
    2. Add the Asset that you created to the Use Case
    3. When viewing the Use Case, the Gradio App should now be visible