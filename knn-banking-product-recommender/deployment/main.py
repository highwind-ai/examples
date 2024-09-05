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
        self.test_df = pd.read_csv(
            "/app/saved_model/df_encoded_test_for_predictions.csv"
        )
        self.train_df = pd.read_csv(
            "/app/saved_model/df_encoded_train_for_predictions.csv"
        )
        self.indices = joblib.load("/app/saved_model/indices.joblib")
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> np.ndarray:
        """Extract user ID and number of recommendations from inference request."""
        infer_input = payload.inputs[0]
        user_id = int(infer_input.data[0])
        num_recommendations = (
            int(infer_input.data[1]) if len(infer_input.data) > 1 else 2
        )
        return np.array([user_id, num_recommendations])

    def predict(self, data: np.ndarray, *args, **kwargs) -> InferResponse:
        """Generate product recommendations for a given user ID."""
        # Unpack the numpy array
        user_id, num_recommendations = data

        # Get the index of user_id in test_df. If the user is not found, it raises an error
        user_index = self.test_df[self.test_df["cust_id"] == user_id].index
        if user_index.empty:
            raise ValueError(f"User ID {user_id} not found in test data.")
        user_index = user_index[0]

        # Get the product ownership of the nearest neighbors
        neighbors_indices = self.indices.flatten()
        neighbors_products = self.train_df.iloc[neighbors_indices]

        # Sum the number of times each product is owned by the neighbors
        product_counts = (
            neighbors_products[self.owns_product_columns]
            .sum()
            .sort_values(ascending=False)
        )

        # Get the products that the user already owns from the training set
        user_product_row = self.train_df[self.train_df["cust_id"] == user_id]
        if user_product_row.empty:
            raise ValueError(f"User ID {user_id} not found in training data.")

        # Create a set of products that the user already owns
        user_products = set(
            user_product_row[self.owns_product_columns].columns[
                user_product_row[self.owns_product_columns].iloc[0] == 1
            ]
        )

        # Remove products that the user already owns from the list of recommended products
        recommended_products = [
            product for product in product_counts.index if product not in user_products
        ]

        # Get the top recommendations based on the requested number
        recommended_products = recommended_products[:num_recommendations]

        # Map product codes to product names
        recommended_product_names = self.map_product_codes_to_names(
            recommended_products
        )

        # Generate response
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
