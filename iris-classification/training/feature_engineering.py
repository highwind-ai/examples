"""Script template containing functions for feature engineering."""

# Library imports
from typing import Tuple
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import joblib


def read_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Read data from a CSV file or any other source.

    Args:
        file_path (str): The path to the data file

    Returns:
        pd.DataFrame: The read data
    """
    # Example: Read data from CSV file
    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.copy()
    y = X.pop("target")

    return X, y


def feature_engineering(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Perform feature engineering steps.

    Args:
        X_train (pd.DataFrame): Training feature matrix

    Returns:
        pd.DataFrame: Scaled training and testing feature matrices
    """

    # Perform feature engineering steps
    # ...

    # Initialise scaler and scale train features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled, scaler
