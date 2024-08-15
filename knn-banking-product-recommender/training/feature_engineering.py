"""Script template containing functions for feature engineering."""

# This script will focus on handle encoding categorical features,
# scaling features and preparing the data for modeling.

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from typing import Tuple, Dict

from utilities import (
    update_gender_columns,
    update_date_id,
    map_products_with_numbers,
    get_binary_item_ownership_columns,
    pre_process_categorical_data,
)


ARTIFACT_SAVE_DIR = "../saved_model/"


def feature_engineering(
    df_train_preprocessed: pd.DataFrame,
    df_test_preprocessed: pd.DataFrame,
    expected_num_items: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, PCA]:
    """
    Perform feature engineering on the preprocessed training and test datasets, including updating
    gender columns, date formatting, product ownership mapping, scaling data and applying PCA.

    Parameters:
    ----------
    df_train_preprocessed : pd.DataFrame
        Preprocessed training data.
    df_test_preprocessed : pd.DataFrame
        Preprocessed test data.
    expected_num_items : int
        The expected number of products to be processed (it starts from 0)


    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, PCA]
        PCA transformed training and test DataFrames, encoded train and test and the scaler and PCA model used.
    """

    retained_columns = [
        "cust_id",
        "date_id",
        "age",
        "gender",
        "emp_index",
        "channel_joined",
        "cust_primary_type",
        "cust_rel_type_at_start_month",
        "activity_index",
        "gross_income",
        "cust_category",
        "deceased_index",
    ]

    product_columns = [
        "savings_acc",
        "guarantees",
        "current_acc",
        "derivada_acc",
        "payroll_acc",
        "jnr_acc",
        "más_particular_acc",
        "particular_account",
        "particular_plus_account",
        "short_term_deposits",
        "medium_term_deposits",
        "long_term_deposits",
        "e_acc",
        "funds",
        "mortgage",
        "pensions_plan",
        "loans",
        "taxes",
        "credit_card",
        "securities",
        "home_acc",
        "payroll",
        "pensions",
        "direct_debit",
    ]

    all_columns = retained_columns + product_columns

    # Subset columns
    df_train_subset_cols = df_train_preprocessed[all_columns]
    df_test_subset_cols_cleaned = df_test_preprocessed.dropna(subset=["gender"])

    # Remove rows with missing values
    df_train_subset_cols_cleaned = df_train_subset_cols.dropna(
        subset=["age", "deceased_index", "gender"]
    )

    # Update gender values
    df_train_subset_cols_cleaned = update_gender_columns(df_train_subset_cols_cleaned)
    df_test_subset_cols_cleaned = update_gender_columns(df_test_subset_cols_cleaned)

    # Update date_id format
    df_train_subset_cols_cleaned = update_date_id(df_train_subset_cols_cleaned)
    df_test_subset_cols_cleaned = update_date_id(df_test_subset_cols_cleaned)

    # Map customer categories
    category_mapping: Dict[str, int] = {
        "02 - PARTICULARES": 2,
        "03 - UNIVERSITARIO": 3,
        "01 - TOP": 1,
    }
    df_train_subset_cols_cleaned["cust_category"] = df_train_subset_cols_cleaned[
        "cust_category"
    ].replace(category_mapping)
    df_test_subset_cols_cleaned["cust_category"] = df_test_subset_cols_cleaned[
        "cust_category"
    ].replace(category_mapping)

    # Drop deceased_index
    df_train_subset_cols_cleaned = df_train_subset_cols_cleaned.drop(
        columns=["deceased_index"]
    )
    df_test_subset_cols_cleaned = df_test_subset_cols_cleaned.drop(
        columns=["deceased_index"]
    )

    # Impute missing product ownership columns in the test set
    product_probabilities: Dict[str, float] = {
        "savings_acc": 0.00010250168731287568,
        "guarantees": 2.320238767254206e-05,
        "current_acc": 0.6562846016619959,
        "derivada_acc": 0.000394440590433215,
        "payroll_acc": 0.08098925582600207,
        "jnr_acc": 0.00947538520369053,
        "más_particular_acc": 0.00971078664317841,
        "particular_account": 0.12918318490658248,
        "particular_plus_account": 0.04336900724913079,
        "short_term_deposits": 0.0017499446373408067,
        "medium_term_deposits": 0.0016640370627304451,
        "long_term_deposits": 0.04304307244255353,
        "e_acc": 0.08282503461267579,
        "funds": 0.018514624259359225,
        "mortgage": 0.005878280861672064,
        "pensions_plan": 0.009185355357783755,
        "loans": 0.00253486085322522,
        "taxes": 0.051892140029640314,
        "credit_card": 0.044463484434722284,
        "securities": 0.02564921161811102,
        "home_acc": 0.003853652261410435,
        "payroll": 0.05475734120608948,
        "pensions": 0.05946419519140795,
        "direct_debit": 0.12811146955699887,
    }

    # Add missing product columns to the test DataFrame
    for product in product_probabilities:
        if product not in df_test_subset_cols_cleaned.columns:
            df_test_subset_cols_cleaned[product] = np.nan

    num_rows: int = len(df_test_subset_cols_cleaned)

    # Fill product columns based on probabilities
    for product, prob in product_probabilities.items():
        if product in df_test_subset_cols_cleaned.columns:
            df_test_subset_cols_cleaned[product] = np.random.binomial(
                1, prob, size=num_rows
            )

    # Ensure all columns are present and in the correct order
    required_columns = [
        "cust_id",
        "date_id",
        "age",
        "gender",
        "emp_index",
        "channel_joined",
        "cust_primary_type",
        "cust_rel_type_at_start_month",
        "activity_index",
        "gross_income",
        "cust_category",
    ] + list(product_probabilities.keys())

    # Ensure the DataFrame has the required columns
    for col in required_columns:
        if col not in df_test_subset_cols_cleaned.columns:
            df_test_subset_cols_cleaned[col] = np.nan

    # Reorder columns to match the required list
    df_test_subset_cols_cleaned = df_test_subset_cols_cleaned[required_columns]

    # Create the 'owned_products' column
    df_train_subset_cols_cleaned["owned_products"] = df_train_subset_cols_cleaned[
        product_columns
    ].apply(lambda row: [col for col in product_columns if row[col] != 0], axis=1)
    df_test_subset_cols_cleaned["owned_products"] = df_test_subset_cols_cleaned[
        product_columns
    ].apply(lambda row: [col for col in product_columns if row[col] != 0], axis=1)

    # Map product names to numbers
    product_mapping = {product: idx for idx, product in enumerate(product_columns)}
    df_train_subset_cols_cleaned["owned_products"] = df_train_subset_cols_cleaned[
        "owned_products"
    ].apply(lambda x: map_products_with_numbers(x, product_mapping))
    df_test_subset_cols_cleaned["owned_products"] = df_test_subset_cols_cleaned[
        "owned_products"
    ].apply(lambda x: map_products_with_numbers(x, product_mapping))

    # Get the latest record for each customer
    df_train_result = df_train_subset_cols_cleaned.loc[
        df_train_subset_cols_cleaned.groupby("cust_id")["date_id"].idxmax()
    ]
    df_test_result = df_test_subset_cols_cleaned.loc[
        df_test_subset_cols_cleaned.groupby("cust_id")["date_id"].idxmax()
    ]

    # Get ownership cols
    ownership_cols_train = get_binary_item_ownership_columns(
        df_=df_train_result,
        owned_items_col="owned_products",
        nb_items=expected_num_items,
    )
    ownership_cols_test = get_binary_item_ownership_columns(
        df_=df_test_result,
        owned_items_col="owned_products",
        nb_items=expected_num_items,
    )

    # Replace old ownership col
    df_train_processed = df_train_result.drop(columns="owned_products").join(
        ownership_cols_train
    )
    df_test_processed = df_test_result.drop(columns="owned_products").join(
        ownership_cols_test
    )

    # Fill NaN ownership with 0s
    df_train_processed = df_train_processed.fillna(
        {col: 0 for col in ownership_cols_train.columns}
    )
    df_test_processed = df_test_processed.fillna(
        {col: 0 for col in ownership_cols_train.columns}
    )

    # Specify columns to one-hot encode
    columns_to_encode = [
        "gender",
        "emp_index",
        "channel_joined",
        "cust_rel_type_at_start_month",
    ]

    # Convert categorical data to numerical data
    df_encoded_train = pre_process_categorical_data(
        df_train_processed, columns_to_encode
    )
    df_encoded_test = pre_process_categorical_data(df_test_processed, columns_to_encode)

    # Drop cust_id and date_id
    df_encoded_train_for_predictions = df_encoded_train.drop(
        ["cust_id", "date_id"], axis=1
    )
    df_encoded_test_for_predictions = df_encoded_test.drop(
        ["cust_id", "date_id"], axis=1
    )

    # Save the DataFrames to CSV files
    df_encoded_train_for_predictions.to_csv(
        os.path.join(ARTIFACT_SAVE_DIR, "train_for_predictions.csv"), index=False
    )
    df_encoded_test_for_predictions.to_csv(
        os.path.join(ARTIFACT_SAVE_DIR, "test_for_predictions.csv"), index=False
    )

    # Select only the numerical columns for PCA
    numerical_cols = df_encoded_train_for_predictions.select_dtypes(
        include=[np.number]
    ).columns

    df_train_numerical = df_encoded_train_for_predictions[numerical_cols]
    df_test_numerical = df_encoded_test_for_predictions[numerical_cols]

    # Standardize the data
    scaler = StandardScaler()
    df_train_scaled = scaler.fit_transform(df_train_numerical)
    df_test_scaled = scaler.transform(df_test_numerical)

    # Apply PCA to keep 95% of the variance
    pca = PCA(n_components=0.95)
    df_train_pca = pca.fit_transform(df_train_scaled)
    df_test_pca = pca.transform(df_test_scaled)

    # Save PCA and scaler results
    joblib.dump(scaler, os.path.join(ARTIFACT_SAVE_DIR, "scaler.joblib"))
    joblib.dump(pca, os.path.join(ARTIFACT_SAVE_DIR, "pca.joblib"))

    return (
        df_train_pca,
        df_test_pca,
        df_encoded_train_for_predictions,
        df_encoded_test_for_predictions,
        scaler,
        pca,
    )
