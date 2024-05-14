"""Script containing functions for data prep prior to training."""

# Add any required imports
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def read_data(dataset_name: str, subset: str) -> DatasetDict:
    # Read data from a dataset using load_dataset.

    # Load OPUS Books dataset
    books = load_dataset(dataset_name, subset)
    print(books.shape)

    subset_train = books["train"].select(range(500))
    subset_test = books["train"].select(range(50))
    books = DatasetDict({"train": subset_train, "test": subset_test})

    return books


# Preprocess the data
def preprocess_data(examples):

    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    source_lang = "en"
    target_lang = "fr"
    prefix = "translate English to French: "

    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]

    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=128, truncation=True
    )
    return model_inputs
