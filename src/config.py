"""
Configuration module for the Haensel AMS Price Prediction project.
Defines file paths, column definitions, and model hyperparameters.
"""
from pathlib import Path

# Base directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

# File paths
RAW_DATA_FILE = DATA_RAW_DIR / "sample.csv"
CLEANED_DATA_FILE = DATA_PROCESSED_DIR / "cleaned_data.csv"
MODEL_SAVE_FILE = ARTIFACTS_DIR / "best_rf_model.joblib"

# Data column specifications
TARGET_COLUMN = "price"
CATEGORICAL_COLUMNS = ["dow"]
NUMERICAL_COLUMNS = ["loc1", "loc2", "para1", "para2", "para3", "para4"]
ALL_FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# Outlier filtering thresholds
PARA1_OUTLIER_THRESHOLD = 10.0

# Train/Test Split & CV parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Hyperparameter search grid for Random Forest
RF_PARAM_GRID = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": ["sqrt", "log2", 1.0],
}
