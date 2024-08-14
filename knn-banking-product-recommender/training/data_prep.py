"""Script containing functions for data pre-processing."""

# The script will focus on loading the data, renaming columns,
# handling data type conversions, dealing with missing values, and updating specific columns.

import pandas as pd
from typing import Tuple
from huggingface_hub import hf_hub_download

REPO_ID = "MelioAI/santander-product-recommendation"
HF_TRAIN_DATASET_NAME = "train_ver2.csv"
HF_TEST_DATASET_NAME = "test_ver2.csv"


def preprocess_data(huggingface_token: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the training and test datasets.

    Args:
        huggingface_token (str): The Hugging Face access token for authentication.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - A cleaned training dataset with columns renamed, data types converted, and missing values handled.
            - A test dataset with columns renamed, data types converted, and missing values handled.
    """

    # Load datasets
    ds_train = pd.read_csv(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=HF_TRAIN_DATASET_NAME,
            repo_type="dataset",
            token=huggingface_token,
        )
    )

    ds_test = pd.read_csv(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=HF_TEST_DATASET_NAME,
            repo_type="dataset",
            token=huggingface_token,
        )
    )

    # Convert column names from Spanish to more readable English names
    col_names = {
        "fecha_dato": "date_id",
        "ncodpers": "cust_id",
        "ind_empleado": "emp_index",
        "pais_residencia": "cust_country_res",
        "sexo": "gender",
        "fecha_alta": "cust_start_date_first_holder_contract",
        "ind_nuevo": "new_cust_index",
        "antiguedad": "cust_seniority",
        "indrel": "cust_primary_type",
        "ult_fec_cli_1t": "cust_last_primary_date",
        "indrel_1mes": "cust_type_at_start_month",
        "tiprel_1mes": "cust_rel_type_at_start_month",
        "indresi": "residence_index",
        "indext": "foreigner_index",
        "conyuemp": "spouse_index",
        "canal_entrada": "channel_joined",
        "indfall": "deceased_index",
        "tipodom": "address_type",
        "cod_prov": "province",
        "nomprov": "province_name",
        "ind_actividad_cliente": "activity_index",
        "renta": "gross_income",
        "segmento": "cust_category",
        "ind_ahor_fin_ult1": "savings_acc",
        "ind_aval_fin_ult1": "guarantees",
        "ind_cco_fin_ult1": "current_acc",
        "ind_cder_fin_ult1": "derivada_acc",
        "ind_cno_fin_ult1": "payroll_acc",
        "ind_ctju_fin_ult1": "jnr_acc",
        "ind_ctma_fin_ult1": "más_particular_acc",
        "ind_ctop_fin_ult1": "particular_account",
        "ind_ctpp_fin_ult1": "particular_plus_account",
        "ind_deco_fin_ult1": "short_term_deposits",
        "ind_deme_fin_ult1": "medium_term_deposits",
        "ind_dela_fin_ult1": "long_term_deposits",
        "ind_ecue_fin_ult1": "e_acc",
        "ind_fond_fin_ult1": "funds",
        "ind_hip_fin_ult1": "mortgage",
        "ind_plan_fin_ult1": "pensions_plan",
        "ind_pres_fin_ult1": "loans",
        "ind_reca_fin_ult1": "taxes",
        "ind_tjcr_fin_ult1": "credit_card",
        "ind_valo_fin_ult1": "securities",
        "ind_viv_fin_ult1": "home_acc",
        "ind_nomina_ult1": "payroll",
        "ind_nom_pens_ult1": "pensions",
        "ind_recibo_ult1": "direct_debit",
    }

    # Rename columns
    ds_train_renamed = ds_train.rename(col_names, axis=1)
    ds_test_renamed = ds_test.rename(col_names, axis=1)

    # Perform data type conversions
    ds_train_renamed["age"] = pd.to_numeric(ds_train_renamed["age"], errors="coerce")
    ds_train_renamed["gross_income"] = pd.to_numeric(
        ds_train_renamed["gross_income"], errors="coerce"
    )
    ds_train_renamed["cust_seniority"] = pd.to_numeric(
        ds_train_renamed["cust_seniority"], errors="coerce"
    )
    ds_train_renamed["cust_start_date_first_holder_contract"] = pd.to_datetime(
        ds_train_renamed["cust_start_date_first_holder_contract"], errors="coerce"
    )
    ds_train_renamed["date_id"] = pd.to_datetime(ds_train_renamed["date_id"])

    ds_test_renamed["age"] = pd.to_numeric(ds_test_renamed["age"], errors="coerce")
    ds_test_renamed["gross_income"] = pd.to_numeric(
        ds_test_renamed["gross_income"], errors="coerce"
    )
    ds_test_renamed["cust_seniority"] = pd.to_numeric(
        ds_test_renamed["cust_seniority"], errors="coerce"
    )
    ds_test_renamed["cust_start_date_first_holder_contract"] = pd.to_datetime(
        ds_test_renamed["cust_start_date_first_holder_contract"], errors="coerce"
    )
    ds_test_renamed["date_id"] = pd.to_datetime(ds_test_renamed["date_id"])

    # Handle missing values
    columns_to_drop = ["cust_last_primary_date", "spouse_index"]
    ds_train_renamed.drop(
        columns=[col for col in columns_to_drop if col in ds_train_renamed.columns],
        axis=1,
        inplace=True,
    )
    ds_test_renamed.drop(
        columns=[col for col in columns_to_drop if col in ds_test_renamed.columns],
        axis=1,
        inplace=True,
    )

    cols = [
        "emp_index",
        "cust_country_res",
        "cust_start_date_first_holder_contract",
        "new_cust_index",
        "cust_primary_type",
        "cust_type_at_start_month",
        "cust_rel_type_at_start_month",
        "province",
        "province_name",
        "activity_index",
        "channel_joined",
        "cust_category",
    ]

    for col in cols:
        if col in ds_train_renamed.columns and col in ds_test_renamed.columns:
            ds_train_renamed[col].fillna(ds_train_renamed[col].mode()[0], inplace=True)
            ds_test_renamed[col].fillna(ds_test_renamed[col].mode()[0], inplace=True)

    ds_train_renamed["gross_income"].fillna(
        ds_train_renamed["gross_income"].mean(), inplace=True
    )
    ds_test_renamed["gross_income"].fillna(
        ds_test_renamed["gross_income"].mean(), inplace=True
    )

    # Drop rows where 'payroll' or 'pensions' is NaN in the training data
    df_train_cleaned = ds_train_renamed.dropna(subset=["payroll", "pensions"])

    return df_train_cleaned, ds_test_renamed
