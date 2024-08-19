### Overview:

The training folder utilizes KFP's [Containerized Python Components](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/containerized-python-components/) and contains the following scripts for this example:

- `data_prep`: Responsible for loading the data, renaming columns, handling data type conversions, dealing with missing values, and updating specific columns.

- `feature_engineering`: Handles encoding categorical features, scaling features, and preparing the data for modeling.

-  `model_training`: Conducts training of a KNN model on PCA-transformed training data.

- `pipeline.py`: Manages KFP pipeline construction logic.

## Usage:
 - Set up your environment with the required dependencies - **If you have not already done so**

 - Run the following commands
    - **NB:** Ensure you are in the root directory before running the commands

    ```shell
    # Use Python 3.9/3.11

    # Initial setup

    python -m venv .train && source .train/bin/activate
    pip install -r training/train-requirements.txt

    # Activate after setup (run every time)

    source .train/bin/activate
    ```

- Navigate within the `training` folder to complile the pipeline script by running the following:

    ```shell
    python -m pipeline
    ```

  - An IR YAML file will be generated.

  - This file will be uploaded to Highwind (Highwind Pipeline is currently under development).


## Highwind Asset and Use Case Creation:

  - Refer to the  [Highwind User Manual](https://docs.highwind.ai/zindi/deploy/) for detailed instructions on creating an asset and use case.
