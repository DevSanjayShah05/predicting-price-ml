"""
Data loading and validation module.
Responsible for loading the raw dataset and ensuring integrity.
"""
import urllib.request
from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_FILE, TARGET_COLUMN, ALL_FEATURE_COLUMNS

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/haensel-ams/recruitment_challenge/"
    "master/ML_201609/sample.csv"
)


def ensure_raw_data(filepath: Path = RAW_DATA_FILE) -> Path:
    """Ensure the raw dataset exists locally; download if not present."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        print(f"Dataset not found at {filepath}. Downloading from upstream...")
        urllib.request.urlretrieve(RAW_DATA_URL, filepath)
        print("Download completed successfully.")
    return filepath


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw dataset from CSV into a pandas DataFrame."""
    ensure_raw_data(filepath)
    df = pd.read_csv(filepath)
    validate_raw_data(df)
    return df


def validate_raw_data(df: pd.DataFrame) -> bool:
    """Validate that the loaded DataFrame matches expected schema."""
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    expected_columns = set(ALL_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return True
