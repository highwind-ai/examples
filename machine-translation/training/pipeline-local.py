from kfp import dsl
from kfp import local
from data_prep import read_data, preprocess_data
from model_evaluation import compute_metrics
from model_training import train_model

local.init(
    runner=local.DockerRunner(), raise_on_error=True, pipeline_root="./_local_test"
)


@dsl.component(
    base_image="python:3.10-bookworm",
    target_image="637423190872.dkr.ecr.eu-west-1.amazonaws.com/highwind/ef620ece-2b04-4f7e-8932-31dd3ba63e03/7a0550a8-b644-4267-b2be-85977ab21ff0:latest",
    packages_to_install=[
        "tensorflow==2.15.0",
        "tf-keras==2.15.0",
        "urllib3==1.26.0  ",
        "requests-toolbelt==0.10.1",
        "transformers==4.39.3",
        "datasets==2.18.0",
        "numpy==1.26.4",
        "yq==3.2.3",
        "docker==7.0.0",
    ],
)
def ml_process(
    model_path: dsl.Output[dsl.Model],
):
    import os
    from transformers import AutoTokenizer
    from transformers import DataCollatorForSeq2Seq
    import tensorflow as tf

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


@dsl.pipeline(
    name="tensorflow-huggingface-machine-translation",
    # description="Example Tensorflow Training Pipeline for Machine Translation between en and fr.",
)
def tensorflow_pipeline() -> dsl.Model:
    ml_op = ml_process().set_display_name("ml-process")
    return ml_op.outputs["model_path"]


# There is currently a bug in KFP where optional arguments will result in an invalid IR YAML
# Make sure that the IR YAML compiled does not have `isOptional: true` contained anywhere

if __name__ == "__main__":
    print("Starting pipeline")
    tensorflow_pipeline()
    print("Done")