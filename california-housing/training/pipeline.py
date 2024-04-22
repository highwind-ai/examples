from kfp import dsl


@dsl.component(
    base_image="python:3.9-bookworm",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.2",
        "urllib3==1.25.11"
    ],
    install_kfp_package=True
)
def train_component(
    input_data_dir: dsl.Input[dsl.Dataset],
    artifact_output_dir: dsl.Output[dsl.Artifact],
    train_data_file_name: str,
    model_output_filename: str,
    scaler_output_filename: str
) -> None:
    # Define imports and constants
    # Because KFP Python Lightweight components must be hermetic
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    import pandas as pd
    import joblib

    RANDOM_SEED = 42
    TARGET_COLUMN = "MedHouseVal"
    MODEL_ARGS = {
        "alpha": 0.01,
        "fit_intercept": True,
        "random_state": RANDOM_SEED
    }

    # Load and process raw training data
    print(f"Creating output dir: {artifact_output_dir.path}")
    os.makedirs(artifact_output_dir.path, exist_ok=True)

    train_data_path = os.path.join(input_data_dir.path, train_data_file_name)
    print(f"Loading train data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    print("Separating features and labels")
    X_train = train_df.copy()
    y_train = X_train.pop(TARGET_COLUMN)

    print("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    scaler_save_path = os.path.join(artifact_output_dir.path, scaler_output_filename)
    print(f"Saving scaler to: {scaler_save_path}")
    joblib.dump(scaler, scaler_save_path)

    # Train model on processed data
    print("Defining and training model")
    model = Lasso(**MODEL_ARGS)
    model.fit(X_train_scaled, y_train)
    print(f"Model trained: {model.__str__()}")
    print(f"Trained model weights: {model.coef_}")

    model_save_path = os.path.join(artifact_output_dir.path, model_output_filename)
    print(f"Saving trained model to: {model_save_path}")
    joblib.dump(model, model_save_path)


@dsl.pipeline(
    name="Train California housing price predictor",
)
def train_pipeline() -> None:
    # Define Highwind Dataset importer
    importer_task = dsl.importer(
        artifact_uri="minio://mlpipeline/v2/artifacts/pipeline/hw-ingestion-pipeline/<paste from HW>", # paste from Highwind
        artifact_class=dsl.Dataset, reimport=False
    )

    # Process Dataset
    data_op = train_component(
        input_data_dir=importer_task.output,
        train_data_file_name="train.csv",
        model_output_filename="model.joblib",
        scaler_output_filename="scaler.joblib"
    )
    data_op.set_display_name("train-model")

# There is currently a bug in KFP where optional arguments will result in an invalid IR YAML
# Make sure that the IR YAML compiled does not have `isOptional: true` contained anywhere

if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(
        train_pipeline,
        package_path="train-pipeline.yaml",
    )
    print("Done")
