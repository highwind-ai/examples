"""Script template containing functions for data prep prior to training."""

# Library imports
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(file_path: str) -> pd.DataFrame:
    """
    Read data from a CSV file or any other source.

    Args:
        file_path (str): The path to the data file

    Returns:
        pd.DataFrame: The read data
    """
    # Example: Read data from CSV file
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the data (data cleaning, handling missing values, etc.).

    Args:
        data (pd.DataFrame): The input data

    Returns:
        pd.DataFrame: The preprocessed data
    """
    # Perform pre-processing steps (handle missing values, remove unnecessary columns, etc.)
    # ...
    # ...
    # ...
    # Add helper functions to the utilities folder eg: feature encoding, null removals to the script becoming too long

    # Example: Drop missing values
    preprocessed_data = data.dropna()

    return preprocessed_data


def split_data(
    X: pd.DataFrame, y: str, test_size: float = 0.2, random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        X (pd.DataFrame): The input data
        y (str): The name of the target column
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is set to 0.2
        random_state (int or None, optional): Seed for random number generation. Default is set to None

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the feature matrix for training,
        the feature matrix for testing, the target variable for training, and the target variable for testing.
    """
    # Perform the required data splitting steps (e.g., train, validate, and test sets)
    # ...
    # ...
    # ...

    # Example: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(y, axis=1),
        X[y],
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
