"""
Unit tests for data_loader module.
"""
import pandas as pd
from src.data_loader import load_raw_data, validate_raw_data
from src.config import TARGET_COLUMN, ALL_FEATURE_COLUMNS


def test_load_raw_data_schema():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET_COLUMN in df.columns
    for col in ALL_FEATURE_COLUMNS:
        assert col in df.columns


def test_validate_raw_data_valid():
    df = pd.DataFrame(
        {
            "loc1": [1, 2],
            "loc2": [10, 20],
            "para1": [1, 1],
            "dow": ["Mon", "Tue"],
            "para2": [100, 200],
            "para3": [1000.0, 2000.0],
            "para4": [5.0, 6.0],
            "price": [150.0, 250.0],
        }
    )
    assert validate_raw_data(df) is True
