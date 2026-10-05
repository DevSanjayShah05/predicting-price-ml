"""
Data preprocessing and feature engineering pipelines.
Ensures zero data leakage by creating strict fit/transform abstractions.
"""
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    TARGET_COLUMN,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    PARA1_OUTLIER_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
    CLEANED_DATA_FILE,
)


def clean_raw_data(
    df: pd.DataFrame,
    filter_para1_outliers: bool = True,
    threshold: float = PARA1_OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    """
    Clean the raw dataset:
    1. Coerce non-numeric string values (e.g. 'S', 'T') in loc1 and loc2 to NaN.
    2. Remove rows with null/missing values.
    3. Filter extreme anomalies in para1 (values >= threshold).
    """
    cleaned = df.copy()

    # Coerce loc1 and loc2 to numeric
    cleaned["loc1"] = pd.to_numeric(cleaned["loc1"], errors="coerce")
    cleaned["loc2"] = pd.to_numeric(cleaned["loc2"], errors="coerce")

    # Drop any resulting or existing NaNs
    cleaned = cleaned.dropna()

    # Filter out extreme para1 outliers if requested
    if filter_para1_outliers:
        cleaned = cleaned[cleaned["para1"] < threshold]

    return cleaned.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and test feature/target sets.
    MUST be called before any feature scaling to prevent data leakage.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def get_scaler(scaling_method: str = "standard"):
    """Return the scikit-learn scaler corresponding to the requested method."""
    method = scaling_method.lower()
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "robust":
        return RobustScaler()
    else:
        raise ValueError(
            f"Unsupported scaling method: {scaling_method}. "
            "Choose from 'standard', 'minmax', or 'robust'."
        )


def build_preprocessor(
    scaling_method: str = "standard",
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerical features and
    one-hot encodes categorical features.
    """
    num_cols = numerical_features or NUMERICAL_COLUMNS
    cat_cols = categorical_features or CATEGORICAL_COLUMNS

    scaler = get_scaler(scaling_method)
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def save_processed_data(df: pd.DataFrame, filepath: Path = CLEANED_DATA_FILE) -> Path:
    """Save the cleaned dataset to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath
