"""
Unit tests for preprocessing module.
"""
import pandas as pd
from src.preprocessing import clean_raw_data, split_data, build_preprocessor, get_scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def test_clean_raw_data_handles_strings_and_outliers():
    raw_sample = pd.DataFrame(
        {
            "loc1": ["0", "S", "2", "3"],
            "loc2": ["01", "T", "20", "30"],
            "para1": [1, 2, 15, 3],  # 15 is an outlier >= 10
            "dow": ["Mon", "Tue", "Wed", "Thu"],
            "para2": [100, 200, 300, 400],
            "para3": [1000.0, 2000.0, 3000.0, 4000.0],
            "para4": [5.0, 6.0, 7.0, 8.0],
            "price": [150.0, 250.0, 350.0, 450.0],
        }
    )

    cleaned = clean_raw_data(raw_sample, filter_para1_outliers=True, threshold=10)

    # Row 1 ('S', 'T') should be coerced to NaN and dropped
    # Row 2 (para1 = 15) should be filtered out
    # Only Row 0 and Row 3 remain
    assert len(cleaned) == 2
    assert (cleaned["para1"] < 10).all()
    assert cleaned["loc1"].dtype in [int, float]
    assert cleaned["loc2"].dtype in [int, float]


def test_split_data_leakage_prevention():
    df = pd.DataFrame(
        {
            "loc1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "loc2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "para1": [1] * 10,
            "dow": ["Mon"] * 10,
            "para2": [100] * 10,
            "para3": [1000.0] * 10,
            "para4": [5.0] * 10,
            "price": [100.0 * i for i in range(1, 11)],
        }
    )

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert "price" not in X_train.columns


def test_get_scaler():
    assert isinstance(get_scaler("standard"), StandardScaler)
    assert isinstance(get_scaler("minmax"), MinMaxScaler)
    assert isinstance(get_scaler("robust"), RobustScaler)


def test_build_preprocessor_transform():
    df = pd.DataFrame(
        {
            "loc1": [1, 2, 3],
            "loc2": [10, 20, 30],
            "para1": [1, 2, 1],
            "dow": ["Mon", "Tue", "Wed"],
            "para2": [100, 200, 300],
            "para3": [1000.0, 2000.0, 3000.0],
            "para4": [5.0, 6.0, 7.0],
        }
    )
    preprocessor = build_preprocessor("standard")
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    # 6 numeric features + 2 dummy features (3 unique days - 1 dropped) = 8
    assert transformed.shape[1] == 8
