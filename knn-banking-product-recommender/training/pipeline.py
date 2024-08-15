"""Script containing the kfp pipeline construction logic"""

from kfp import dsl
from data_prep import preprocess_data
from feature_engineering import feature_engineering


@dsl.component(
    base_image="python:3.10-bookworm",
    target_image="637423190872.dkr.ecr.eu-west-1.amazonaws.com/highwind/ef620ece-2b04-4f7e-8932-31dd3ba63e03/ff127876-449c-4036-91d7-af54ebfc5af7:latest",
    packages_to_install=[
        "numpy",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.2",
    ],
    install_kfp_package=True,
)
def ml_process(
    artifact_output_dir: dsl.Output[dsl.Artifact],
    access_token: str,
    train_output_filename: str,
    test_output_filename: str,
    pca_output_filename: str,
    scaler_output_filename: str,
) -> None:

    # Define imports and constants
    import os
    import pandas as pd

    # Load and pre-pocess raw data
    print(f"Creating output dir: {artifact_output_dir.path}")
    os.makedirs(artifact_output_dir.path, exist_ok=True)

    print("Starting loading and pre-processing datasets from HuggingFace....")
    df_train_cleaned, ds_test_renamed = preprocess_data(access_token)
    print("Data preprocessing complete.")

    print("Starting feature engineering process....")
    df_train_pca, df_test_pca, df_train_encoded, df_test_encoded, scaler, pca = (
        feature_engineering(df_train_cleaned, ds_test_renamed, 25)
    )

    train_save_path = os.path.join(artifact_output_dir.path, train_output_filename)
    print(f"Saving enocded train set to: {train_save_path}")
    joblib.dump(df_train_encoded, train_save_path)

    test_save_path = os.path.join(artifact_output_dir.path, test_output_filename)
    print(f"Saving encoded test set to: {test_save_path}")
    joblib.dump(df_test_encoded, test_save_path)

    scaler_save_path = os.path.join(artifact_output_dir.path, scaler_output_filename)
    print(f"Saving scaler to: {scaler_save_path}")
    joblib.dump(scaler, scaler_save_path)

    pca_save_path = os.path.join(artifact_output_dir.path, pca_output_filename)
    print(f"Saving pca to: {pca_save_path}")
    joblib.dump(pca, pca_save_path)

    print("Feature engineering processing and saving complete.")


@dsl.pipeline(
    name="santander-knn-product-recommendation",
    # NB: Please do not add a "description" to avoid upload issues with the generated IR YAML file
    # description="Example Tensorflow Training Pipeline for Machine Translation between en and fr.",
)
def knn_pipeline() -> None:
    ml_op = (
        ml_process(
            access_token="",  # add access_token from HuggingFace
            train_output_filename="train_for_predictions.csv",
            test_output_filename="test_for_predictions.csv",
            pca_output_filename="pca.joblib",
            scaler_output_filename="scaler.joblib",
        )
        .set_display_name("ml-process")
        .set_memory_request("6G")
        .set_memory_limit("6G")
    )


# There is currently a bug in KFP where optional arguments will result in an invalid IR YAML
# Make sure that the IR YAML compiled does not have `isOptional: true` contained anywhere

if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(
        knn_pipeline,
        package_path="santander_knn_banking_recommendation_pipeline.yaml",
    )
    print("Done")