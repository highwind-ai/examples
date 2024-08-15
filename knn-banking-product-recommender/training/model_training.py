"""Script template containing functions for model training."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

ARTIFACT_SAVE_DIR = "../saved_model/"


def train_knn_model(
    df_train_pca: pd.DataFrame,
    df_test_pca: pd.DataFrame,
    df_encoded_test: pd.DataFrame,
    best_n_neighbors: int,
) -> Tuple[pd.DataFrame, NearestNeighbors, np.ndarray, np.ndarray]:
    """
    Train a KNN model on PCA-transformed training data and save the model and nearest neighbors' distances and indices.

    Parameters:
    ----------
    df_train_pca : pd.DataFrame
        PCA-transformed training data.
    df_test_pca : pd.DataFrame
        PCA-transformed test data.
    df_encoded_test : pd.DataFrame
        DataFrame containing the test data with 'cust_id' column to be saved.
    best_n_neighbors : int
        The number of neighbors to use for the KNN model.

    Returns:
    -------
    Tuple[pd.DataFrame, NearestNeighbors, np.ndarray, np.ndarray]
        A DataFrame containing the test IDs, the trained KNN model, distances, and indices.
    """

    # Initialize KNN
    knn = NearestNeighbors(metric="minkowski", p=2, algorithm="ball_tree")

    # Fit KNN model on the PCA transformed training data
    knn.fit(df_train_pca)

    # Save the model for later use
    save_model_path = os.path.join(ARTIFACT_SAVE_DIR, "model.joblib")
    joblib.dump(knn, save_model_path)

    # Get the nearest neighbors for the test data
    # NB: This can take over an hour and a half to complete running
    distances, indices = knn.kneighbors(df_test_pca, best_n_neighbors)

    # Define file paths
    save_distances_path = os.path.join(ARTIFACT_SAVE_DIR, "distances.joblib")
    save_indices_path = os.path.join(ARTIFACT_SAVE_DIR, "indices.joblib")

    # Save distances and indices
    joblib.dump(distances, save_distances_path)
    joblib.dump(indices, save_indices_path)

    # Ensure cust_id is retained
    df_encoded_test_ids = df_encoded_test[["cust_id"]].reset_index(drop=True)

    # Path to save the CSV file
    test_ids_csv_path = os.path.join(ARTIFACT_SAVE_DIR, "df_encoded_test_ids.csv")

    # Save the DataFrame to a CSV file
    df_encoded_test_ids.to_csv(test_ids_csv_path, index=False)

    return df_encoded_test_ids, knn, distances, indices
