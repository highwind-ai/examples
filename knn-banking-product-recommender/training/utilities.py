"""Script containing helper functions"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


def update_gender_columns(
    df: pd.DataFrame, column_name: str = "gender"
) -> pd.DataFrame:
    """
    Update gender columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the gender column.
    column_name (str): The name of the gender column. Default is 'gender'.

    Returns:
    pd.DataFrame: The DataFrame with updated gender values.
    """
    df[column_name].replace({"H": "M", "V": "F"}, inplace=True)
    return df


def update_date_id(df: pd.DataFrame, column_name: str = "date_id") -> pd.DataFrame:
    """
    Convert the date_id column in the DataFrame to YYYYMMDD format.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the date_id column.
    column_name (str): The name of the date_id column. Default is 'date_id'.

    Returns:
    pd.DataFrame: The DataFrame with date_id values in YYYYMMDD format.
    """
    # Convert column to string
    df[column_name] = df[column_name].astype(str)

    # Convert ISO 8601 (YYYY-MM-DD) to YYYYMMDD
    df[column_name] = pd.to_datetime(
        df[column_name], format="%Y-%m-%d", errors="coerce"
    ).dt.strftime("%Y%m%d")

    return df


def map_products_with_numbers(
    owned_products: List[str], product_mapping: Dict[str, int]
) -> List[int]:
    """
    Replace product names with corresponding numbers from the product_mapping dictionary.
    Return a list with -1 if the list is empty.

    Parameters:
    owned_products (List[str]): List of product names to be replaced.
    product_mapping (Dict[str, int]): Dictionary mapping product names to numbers.

    Returns:
    List[int]: List of corresponding product numbers, or [-1] if input list is empty.
    """
    if not owned_products:
        return [-1]
    return [product_mapping.get(product, -1) for product in owned_products]


def get_customer_details(df: pd.DataFrame, customer_id: int) -> Optional[pd.DataFrame]:
    """
    Retrieve the details of a customer given their customer ID.

    Parameters:
    df (pd.DataFrame): DataFrame containing customer data with columns 'cust_id', 'date_id', and 'owned_products'.
    customer_id (int): The ID of the customer to retrieve details for.

    Returns:
    Optional[pd.DataFrame]: A DataFrame containing the customer's details or an empty DataFrame if the customer ID is not found.
    """
    # Filter the DataFrame to get the row for the given customer_id
    customer_details = df[df["cust_id"] == customer_id]

    if not customer_details.empty:
        # Return the relevant columns as a DataFrame
        return customer_details[["cust_id", "date_id", "owned_products"]]
    else:
        # Return an empty DataFrame if the customer ID is not found
        return pd.DataFrame(columns=["cust_id", "date_id", "owned_products"])


def check_columns(train_columns, test_columns):
    """
    Check if the columns in two sets are the same and print the common and different columns.

    Parameters:
    - train_columns: Set of columns from the training set
    - test_columns: Set of columns from the test set
    """
    if set(train_columns) == set(test_columns):
        print("Train and test sets have the same columns.")
        # common_columns = set(train_columns)
        # print("Common columns:")
        # for col in common_columns:
        #    print(f"- {col}")
    else:
        different_columns_train = set(train_columns) - set(test_columns)
        different_columns_test = set(test_columns) - set(train_columns)

        print("Train and test sets have different columns.")

        if different_columns_train:
            print("Columns present in train but not in test:")
            for col in different_columns_train:
                print(f"- {col}")

        if different_columns_test:
            print("Columns present in test but not in train:")
            for col in different_columns_test:
                print(f"- {col}")


def pre_process_categorical_data(
    data: pd.DataFrame, columns_to_encode: List[str]
) -> pd.DataFrame:
    """
    Convert categorical data to numerical data using one-hot encoding.

    Parameters:
    - data: DataFrame
    - columns_to_encode: List of columns to one-hot encode

    Returns:
    - DataFrame with one-hot encoded columns
    """
    return pd.get_dummies(data, columns=columns_to_encode)


def get_binary_item_ownership_columns(
    df_: pd.DataFrame, owned_items_col: str = "owned_products", nb_items: int = 25
) -> pd.DataFrame:
    """
    Takes a dataframe df_, looks at the owned_product col which contains a list of item IDs,
    and converts that into a set of nb_items binary ownership columns called `owns_product_<ID>`
    which. These ownership indicator columns are returned.
    """

    # Insert dummy user that owns everything
    dummy_user_id = -999
    dummy_user = pd.DataFrame(
        data={owned_items_col: {dummy_user_id: list(range(nb_items))}}
    )
    df_ = pd.concat([df_, dummy_user], ignore_index=False)

    # Explode the owned items column
    df_exploded = df_.explode(owned_items_col)

    # Get dummy columns for ownership
    ownership_cols = pd.get_dummies(
        df_exploded[owned_items_col].astype(int), prefix="owns_product"
    )

    # Group by the original index and sum the dummy columns
    ownership_cols = ownership_cols.groupby(level=0).sum()

    # Drop -1 (none-item) if exists
    if "owns_product_-1" in ownership_cols.columns:
        ownership_cols = ownership_cols.drop(columns=["owns_product_-1"])

    # Remove dummy user
    ownership_cols = ownership_cols.drop(index=dummy_user_id)

    return ownership_cols


def find_best_n_neighbors(
    X: np.ndarray, min_neighbors: int = 1, max_neighbors: int = 10
) -> int:
    """
    Find the best number of neighbors for the K-Nearest Neighbors algorithm by evaluating different values
    using silhouette scores.

    Parameters:
    X (np.ndarray): The input data as a NumPy array.
    min_neighbors (int): The minimum number of neighbors to consider. Default is 1.
    max_neighbors (int): The maximum number of neighbors to consider. Default is 10.

    Returns:
    int: The best number of neighbors based on silhouette scores.
    """
    best_n_neighbors = min_neighbors
    best_score = -1

    for n_neighbors in range(min_neighbors, max_neighbors + 1):
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            knn.fit(X_train)
            distances, indices = knn.kneighbors(X_test)
            if len(X_test) > 1:
                score = silhouette_score(X_test, indices[:, 0])
                scores.append(score)

        mean_score = np.mean(scores)
        # print(f"n_neighbors = {n_neighbors}, Mean Silhouette Score = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_n_neighbors = n_neighbors

    return best_n_neighbors


def get_product_recommendations(
    user_index: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    indices: np.ndarray,
    product_columns: List[str],
    n_recommendations: int = 2,
) -> List[str]:
    """
    Generate product recommendations for a given user based on nearest neighbors.

    Parameters:
    user_index (int): The index of the user in the test set.
    train_df (pd.DataFrame): The DataFrame with training data.
    test_df (pd.DataFrame): The DataFrame with test data.
    indices (np.ndarray): Indices of the nearest neighbors.
    product_columns (List[str]): List of product columns in the DataFrame.
    n_recommendations (int): Number of products to recommend.

    Returns:
    List[str]: List of recommended product names.
    """
    # Get the indices of nearest neighbors
    neighbors_indices = indices[user_index]

    # Get the products owned by the nearest neighbors
    neighbors_products = train_df.iloc[neighbors_indices]

    # Sum the number of times each product is owned by the neighbors
    product_counts = (
        neighbors_products[product_columns].sum().sort_values(ascending=False)
    )

    # Get the products that the user already owns
    user_products = set(
        test_df.loc[user_index, product_columns].index[
            test_df.loc[user_index, product_columns] > 0
        ]
    )

    # Ensure `owns_product_24` is not in recommendations
    if "owns_product_24" in product_counts.index:
        product_counts = product_counts.drop("owns_product_24")

    # Exclude products the user already owns
    recommended_products = [
        product for product in product_counts.index if product not in user_products
    ]

    return recommended_products[:n_recommendations]


def map_product_codes_to_names(
    product_codes: List[str], mapping: Dict[str, str]
) -> List[str]:
    """
    Map product codes to product names.

    Parameters:
    product_codes (List[str]): List of product codes to be mapped.
    mapping (Dict[str, str]): Dictionary mapping product codes to product names.

    Returns:
    List[str]: List of product names.
    """
    return [mapping.get(code, "None") for code in product_codes]


def get_product_recommendations_for_user_id(
    user_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_df_ids: pd.DataFrame,
    indices: np.ndarray,
    product_columns: List[str],
    n_recommendations: int = 2,
) -> List[str]:
    """
    Generate product recommendations for a specific user ID.

    Parameters:
    user_id (int): The user ID for which recommendations are to be generated.
    train_df (pd.DataFrame): The DataFrame with training data.
    test_df (pd.DataFrame): The DataFrame with test data.
    test_df_ids (pd.DataFrame): The DataFrame containing user IDs.
    indices (np.ndarray): Indices of the nearest neighbors.
    product_columns (List[str]): List of product columns in the DataFrame.
    n_recommendations (int): Number of products to recommend.

    Returns:
    List[str]: List of recommended product names.
    """
    # Find the index of the user_id in the test DataFrame
    user_index = test_df_ids[test_df_ids["cust_id"] == user_id].index
    if user_index.empty:
        raise ValueError(f"User ID {user_id} not found in test DataFrame.")

    user_index = user_index[0]  # Get the first index (should be a single index)

    # Get the indices of nearest neighbors
    neighbors_indices = indices[user_index]

    # Get the products owned by the nearest neighbors
    neighbors_products = train_df.iloc[neighbors_indices]

    # Sum the number of times each product is owned by the neighbors
    product_counts = (
        neighbors_products[product_columns].sum().sort_values(ascending=False)
    )

    # Get the products that the user already owns
    user_products = set(
        test_df.loc[user_index, product_columns].index[
            test_df.loc[user_index, product_columns] > 0
        ]
    )

    # Ensure `owns_product_24` is not in recommendations
    if "owns_product_24" in product_counts.index:
        product_counts = product_counts.drop("owns_product_24")

    # Exclude products the user already owns
    recommended_products = [
        product for product in product_counts.index if product not in user_products
    ]

    return recommended_products[:n_recommendations]
