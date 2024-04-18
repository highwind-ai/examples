from kfp import dsl
from feature_engineering import read_data, feature_engineering

MODEL_ARGS = {
    "C": 1,
    "solver": "lbfgs",
    "fit_intercept": True,
    "random_state": 42,
}


@dsl.component(
    base_image="python:3.9-bookworm",
    target_image="637423190872.dkr.ecr.eu-west-1.amazonaws.com/highwind/386cfa8b-bef8-4555-9eaa-583e1e1c9ac0/bbcc9c05-8eb9-4934-b611-06129a5a7c41:v2",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "joblib==1.3.2", "urllib3==1.25.11"],
    install_kfp_package=True
)
def process_data(
    input_data_dir: dsl.Input[dsl.Dataset],
    artifact_output_dir: dsl.Output[dsl.Artifact],
    train_data_file_name: str,
    scaled_data_output_filename: str,
    scaler_output_filename: str
    
) -> None:
    import os
    import pandas as pd
    import joblib

    # Read train data
    X_train, y_train = read_data(os.path.join(input_data_dir.path, train_data_file_name))

    # Feature engineering
    X_train_scaled, scaler = feature_engineering(X_train)

    # Convert to pandas DataFrame
    X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)

    # Rejoin train data
    X_train_scaled["target"] = y_train

    # Make output dirs
    os.makedirs(artifact_output_dir.path, exist_ok=True)

    # Save artifacts
    X_train_scaled.to_csv(
        os.path.join(artifact_output_dir.path, scaled_data_output_filename),
        index=False
    )
    joblib.dump(
        scaler,
        os.path.join(artifact_output_dir.path, scaler_output_filename)
    )


@dsl.component(
    base_image="python:3.9-bookworm",
    target_image="637423190872.dkr.ecr.eu-west-1.amazonaws.com/highwind/386cfa8b-bef8-4555-9eaa-583e1e1c9ac0/bbcc9c05-8eb9-4934-b611-06129a5a7c41:v2",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "joblib==1.3.2", "urllib3==1.25.11"],
    install_kfp_package=True
)
def train_model(
    data_artifact_dir: dsl.Input[dsl.Artifact],
    scaled_data_file_name:str,
    model_output_dir: dsl.Output[dsl.Model],
    model_output_filename:str,
    model_args: dict,
) -> None:
    import os
    import joblib
    from sklearn.linear_model import LogisticRegression

    # Load scaled data
    X_train_scaled, y_train = read_data(os.path.join(data_artifact_dir.path, scaled_data_file_name))

    # Define and train model
    model = LogisticRegression(**model_args)
    model.fit(X_train_scaled, y_train)

    # Save the model for later use
    os.makedirs(model_output_dir.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_output_dir.path, model_output_filename))


@dsl.pipeline(
    name="Train iris classifier",
)
def train_pipeline() -> None:
    # Define Highwind Dataset importer
    importer_task = dsl.importer(
        artifact_uri="minio://mlpipeline/v2/artifacts/pipeline/hw-ingestion-pipeline/2e06de2f-e18b-453f-9a24-fbdbbe084283/download-data/output_path", # paste from Highwind
        artifact_class=dsl.Dataset, reimport=False
    )

    # Process Dataset
    data_op = process_data(
        input_data_dir=importer_task.output,
        train_data_file_name="train.csv",
        scaled_data_output_filename="scaled_train.csv",
        scaler_output_filename="scaler.joblib"
    )
    data_op.set_display_name("process-data")

    # Train model
    train_op = train_model(
        data_artifact_dir=data_op.outputs["artifact_output_dir"],
        scaled_data_file_name="scaled_train.csv",
        model_output_filename="model.joblib",
        model_args=MODEL_ARGS,
    )
    train_op.set_display_name("train-model")


# There is currently a bug in KFP where optional arguments will result in an invalid IR YAML
# Make sure that the IR YAML compiled does not have `isOptional: true` contained anywhere

if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(
        train_pipeline,
        package_path="iris-classifier-train-pipeline.yaml",
    )
    print("Done")
