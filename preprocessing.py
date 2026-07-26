"""
preprocessing.py
----------------
Phase 1 - Data Preprocessing
Customer Churn Prediction & Business Intelligence System

Dataset: IBM Telco Customer Churn (or equivalent, >=5000 records)
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place the raw CSV file at: data/telco_churn_raw.csv

This script handles:
    1. Loading & inspecting the dataset
    2. Handling missing values
    3. Removing duplicate records
    4. Detecting & treating outliers
    5. Feature engineering
    6. Encoding categorical variables
    7. Scaling numerical features
    8. Train/test split
    9. Saving processed artifacts (CSV + encoders + scaler) for later phases
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RAW_DATA_PATH = "data/customer_churn.csv"          # input dataset
PROCESSED_DATA_PATH = "data/telco_churn_processed.csv"  # cleaned dataset (for EDA/dashboard)
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ENCODERS_PATH = "models/encoders.joblib"
SCALER_PATH = "models/scaler.joblib"
TARGET_COL = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# --------------------------------------------------------------------------
# 1. LOAD & INSPECT DATA
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV into a DataFrame and print a quick inspection summary."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Download the IBM Telco Customer Churn dataset and place it there."
        )

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
    print("\nMissing values per column:\n", df.isnull().sum())
    print("=" * 60)

    return df


# --------------------------------------------------------------------------
# 2. HANDLE MISSING VALUES
# --------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Telco dataset quirk: 'TotalCharges' is loaded as object dtype because
    a handful of rows contain blank strings (new customers, tenure=0).
    We coerce to numeric, then impute.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Numerical columns -> fill with median (robust to skew/outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns -> fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[Missing Values] Remaining nulls after imputation: {df.isnull().sum().sum()}")
    return df


# --------------------------------------------------------------------------
# 3. REMOVE DUPLICATES
# --------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset="customerID", keep="first")
    after = len(df)
    print(f"[Duplicates] Removed {before - after} duplicate rows.")
    return df


# --------------------------------------------------------------------------
# 4. OUTLIER DETECTION & TREATMENT
# --------------------------------------------------------------------------
def treat_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    IQR-based capping (winsorization) instead of dropping rows,
    so we don't lose valuable churn signal.
    """
    df = df.copy()
    if cols is None:
        cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cols = [c for c in cols if c in df.columns]

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[Outliers] '{col}': capped {n_outliers} outlier values "
              f"(bounds: {lower:.2f} - {upper:.2f}).")

    return df


# --------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets -> useful for BI dashboard "Tenure Range" filter
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 60, np.inf],
            labels=["0-12", "13-24", "25-48", "49-60", "60+"],
        )

    # Average charge per tenure month (helps detect early high-spenders)
    if {"TotalCharges", "tenure"}.issubset(df.columns):
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Count of subscribed add-on services -> proxy for "engagement"
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    service_cols = [c for c in service_cols if c in df.columns]
    if service_cols:
        df["num_services"] = df[service_cols].apply(
            lambda row: sum(str(v) == "Yes" for v in row), axis=1
        )

    # Flag: has any internet-based add-on but no internet service (data-consistency helper)
    if "InternetService" in df.columns:
        df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # Drop non-predictive identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    print(f"[Feature Engineering] New shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# 6. ENCODE CATEGORICAL VARIABLES
# --------------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame, target_col: str = TARGET_COL):
    df = df.copy()
    encoders = {}

    # Target: Yes/No -> 1/0
    # NOTE: don't gate this on dtype == "object" - depending on the pandas
    # version/backend, a text column can load as "object" OR the newer
    # "string" dtype. Checking the actual values (not the dtype) makes this
    # robust either way, and avoids silently leaving Yes/No un-encoded.
    if target_col in df.columns:
        unique_vals = set(df[target_col].astype(str).str.strip().unique())
        if unique_vals.issubset({"Yes", "No", "1", "0"}):
            df[target_col] = (
                df[target_col].astype(str).str.strip().map(
                    {"Yes": 1, "No": 0, "1": 1, "0": 0}
                )
            )
        df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype(int)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_cols = [c for c in cat_cols if c != target_col]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"[Encoding] Encoded {len(cat_cols)} categorical columns: {list(cat_cols)}")
    return df, encoders


# --------------------------------------------------------------------------
# 7. SCALE NUMERICAL FEATURES
# --------------------------------------------------------------------------
def scale_numerical(df: pd.DataFrame, target_col: str = TARGET_COL):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != target_col]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print(f"[Scaling] Scaled {len(num_cols)} numerical columns.")
    return df, scaler


# --------------------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# --------------------------------------------------------------------------
def split_data(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[Split] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def run_preprocessing_pipeline():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Load
    df = load_data(RAW_DATA_PATH)

    # Keep an untouched copy (pre-encoding) for EDA / dashboard use
    df_clean_for_eda = df.copy()

    # Step 2-5: Clean + engineer
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = treat_outliers(df)
    df = engineer_features(df)

    # Save a human-readable processed dataset (before encoding/scaling)
    # This is what powers Phase 2 EDA and the Phase 6 BI dashboard filters.
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[Saved] Processed (readable) dataset -> {PROCESSED_DATA_PATH}")

    # Step 6-7: Encode + scale (this version feeds the ML models in Phase 3)
    df_encoded, encoders = encode_categorical(df)
    df_final, scaler = scale_numerical(df_encoded)

    # Step 8: Split
    X_train, X_test, y_train, y_test = split_data(df_final)

    # Persist train/test sets
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"[Saved] {TRAIN_PATH}, {TEST_PATH}")

    # Persist encoders & scaler so predict.py / app.py can reuse them
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Saved] {ENCODERS_PATH}, {SCALER_PATH}")

    print("\n✅ Phase 1 (Preprocessing) complete.")
    return {
        "df_clean_for_eda": df_clean_for_eda,
        "df_processed": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()