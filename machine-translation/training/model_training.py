"""Script containing functions for model training."""

# Add any required imports
import os
from transformers import AutoTokenizer, AdamWeightDecay, TFAutoModelForSeq2SeqLM


def train_model(tokenized_books_train, tokenized_books_test, data_collator, model_path):

    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Training
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    tf_train_set = model.prepare_tf_dataset(
        tokenized_books_train,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_books_test,
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=1)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    return model_path
