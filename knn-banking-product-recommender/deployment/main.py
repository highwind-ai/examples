"""Kserve inference script."""

import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List

from kserve import (
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    model_server,
)
from kserve.utils.utils import generate_uuid


class ProductRecommendationModel(Model):
    """Kserve inference implementation of the product recommendation model."""

    def __init__(self, name: str):
        """Initialise model."""
        super().__init__(name)
        self.name = name
        self.train_df = None
        self.test_df = None
        self.test_df_ids = None
        self.indices = None
        self.owns_product_columns = [
            "owns_product_0",
            "owns_product_1",
            "owns_product_2",
            "owns_product_3",
            "owns_product_4",
            "owns_product_5",
            "owns_product_6",
            "owns_product_7",
            "owns_product_8",
            "owns_product_9",
            "owns_product_10",
            "owns_product_11",
            "owns_product_12",
            "owns_product_13",
            "owns_product_14",
            "owns_product_15",
            "owns_product_16",
            "owns_product_17",
            "owns_product_18",
            "owns_product_19",
            "owns_product_20",
            "owns_product_21",
            "owns_product_22",
            "owns_product_23",
            "owns_product_24",
        ]
        self.product_mapping_naming = {
            "owns_product_0": "savings_acc",
            "owns_product_1": "guarantees",
            "owns_product_2": "current_acc",
            "owns_product_3": "derivada_acc",
            "owns_product_4": "payroll_acc",
            "owns_product_5": "jnr_acc",
            "owns_product_6": "más_particular_acc",
            "owns_product_7": "particular_account",
            "owns_product_8": "particular_plus_account",
            "owns_product_9": "short_term_deposits",
            "owns_product_10": "medium_term_deposits",
            "owns_product_11": "long_term_deposits",
            "owns_product_12": "e_acc",
            "owns_product_13": "funds",
            "owns_product_14": "mortgage",
            "owns_product_15": "pensions_plan",
            "owns_product_16": "loans",
            "owns_product_17": "taxes",
            "owns_product_18": "credit_card",
            "owns_product_19": "securities",
            "owns_product_20": "home_acc",
            "owns_product_21": "payroll",
            "owns_product_22": "pensions",
            "owns_product_23": "direct_debit",
            "owns_product_24": "no_product_owned",
        }
        self.ready = False
        self.load()

    def load(self):
        """Load preprocessed data and indices from disk."""
        self.train_df = pd.read_csv(
            "/app/saved_model/df_encoded_test_for_predictions.csv"
        )
        self.test_df = pd.read_csv(
            "/app/saved_model/df_encoded_train_for_predictions.csv"
        )
        self.test_df_ids = pd.read_csv("/app/saved_model/df_encoded_test_ids.csv")
        self.indices = joblib.load("/app/saved_model/indices.joblib")
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> int:
        """Extract user ID from inference request."""
        infer_input = payload.inputs[0]
        user_id = int(infer_input.data[0])
        return user_id

    def predict(self, user_id: int, *args, **kwargs) -> InferResponse:
        """Generate product recommendations for a given user ID."""
        user_index = self.test_df_ids[self.test_df_ids["cust_id"] == user_id].index
        if user_index.empty:
            raise ValueError(f"User ID {user_id} not found in test DataFrame.")

        user_index = user_index[0]
        neighbors_indices = self.indices[user_index]
        neighbors_products = self.train_df.iloc[neighbors_indices]
        product_counts = (
            neighbors_products[self.owns_product_columns]
            .sum()
            .sort_values(ascending=False)
        )
        user_products = set(
            self.test_df.loc[user_index, self.owns_product_columns].index[
                self.test_df.loc[user_index, self.owns_product_columns] > 0
            ]
        )

        if "owns_product_24" in product_counts.index:
            product_counts = product_counts.drop("owns_product_24")

        recommended_products = [
            product for product in product_counts.index if product not in user_products
        ]
        recommended_products = recommended_products[:2]
        recommended_product_names = self.map_product_codes_to_names(
            recommended_products
        )

        response_id = generate_uuid()
        infer_output = InferOutput(
            name="output-0",
            shape=[len(recommended_product_names)],
            datatype="BYTES",
            data=recommended_product_names,
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response

    def map_product_codes_to_names(self, product_codes: List[str]) -> List[str]:
        """Map product codes to product names."""
        return [self.product_mapping_naming.get(code, "None") for code in product_codes]


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name",
    default="product-recommendation-model",
    help="The name that the model is served under.",
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = ProductRecommendationModel(args.model_name)
    ModelServer().start([model])
