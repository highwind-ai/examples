"""Script containing the kfp pipeline construction logic"""

from kfp import dsl
from data_prep import preprocess_data


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
    # model_output_filename: str,
    # knn_indices_output_filename: str,
) -> None:

    # Define imports and constants
    import os
    import pandas as pd

    # Load and pre-pocess raw data
    print(f"Creating output dir: {artifact_output_dir.path}")
    os.makedirs(artifact_output_dir.path, exist_ok=True)

    print("Starting loading and pre-processing datasets from HuggingFace")
    df_train_cleaned, ds_test_renamed = preprocess_data(access_token)

    # Save processed data
    df_train_cleaned_path = os.path.join(artifact_output_dir.path, "train_cleaned.csv")
    ds_test_renamed_path = os.path.join(artifact_output_dir.path, "test_data.csv")

    df_train_cleaned.to_csv(df_train_cleaned_path, index=False)
    ds_test_renamed.to_csv(ds_test_renamed_path, index=False)

    print("Data preprocessing and saving complete.")


"""  train_data_path = os.path.join(input_data_dir.path, train_data_file_name)
    print(f"Loading train data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    test_data_path = os.path.join(input_data_dir.path, test_data_file_name)
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(train_data_path)

    # Stopped here

    os.makedirs(model_path.path, exist_ok=True)

    # Load OPUS Books dataset
    books = read_data("opus_books", "en-fr")

    # Pre-process the data
    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenize the dataset
    tokenized_books = books.map(preprocess_data, batched=True)

    # Create data collator for seq2seq model
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=checkpoint, return_tensors="tf"
    )

    # Train the model
    trained_model_path = train_model(
        tokenized_books["train"],
        tokenized_books["test"],
        data_collator,
        model_path.path,
    )
    """


@dsl.pipeline(
    name="santander-knn-product-recommendation",
    # NB: Please do not add a "description" to avoid upload issues with the generated IR YAML file
    # description="Example Tensorflow Training Pipeline for Machine Translation between en and fr.",
)
def knn_pipeline() -> None:
    ml_op = (
        ml_process(
            access_token="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",
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
