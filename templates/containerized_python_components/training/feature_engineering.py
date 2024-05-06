"""Script template containing functions for feature engineering."""

# Library imports
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_engineering(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature engineering steps.

    Args:
        X_train (pd.DataFrame): Training feature matrix
        X_test (pd.DataFrame): Testing feature matrix

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the scaled training and testing feature matrices
    """

    # Perform feature engineering steps
    # ...

    # Example: Scale or normalize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
