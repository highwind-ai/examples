"""Script template containing functions for model evaluation."""

# Library imports
from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluate_model(
    trained_model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate a trained machine learning model.

    Args:
        trained_model (RandomForestClassifier): Trained machine learning model
        X_test (pd.DataFrame): Testing feature matrix
        y_test (pd.Series): Testing labels

    Returns:
        float: Accuracy on the test set
    """
    # Perform model evaluation steps
    # ...
    # ...
    # ...

    # Example: Evaluate the model using accuracy metric
    # Make predictions on the test set
    predictions = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy
