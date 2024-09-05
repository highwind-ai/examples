# Santander-Product-Recommendation

A banking product recommender using a simple knn model.

## Usage

### Running notebooks and virtual environments

 - Set up your environment with the required dependencies.
 - Run the following commands

    ```shell
    # Use Python 3.9

    # Initial setup
    python -m venv .santander && source .santander/bin/activate
    pip install -r requirements.txt

    # Activate after setup (run every time)
    source .santander/bin/activate
    ```

 - Use ipykernel to create a new kernel associated with the virtual environment:

    ```shell
      python -m ipykernel install --user --name=.santander --display-name "Python (.santander)"

    ```

### Model serving and inference

-  Set up the serve environment by running the following:
   ```shell
   python -m venv .serve.venv && source .serve.venv/bin/activate
   pip install -r deployment/serve-requirements.txt
   ```

-  Activate after setup (run every time)
   ```shell
   source .serve.venv/bin/activate
   ```

