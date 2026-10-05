"""
Model definition, training, pipeline assembly, and Grid Search tuning.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    MODEL_SAVE_FILE,
)


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Return a catalog of candidate regression models:
    - Naive Baseline (Mean Regressor)
    - Linear Models (OLS, Ridge, Lasso)
    - Decision Tree & Ensemble Models (Random Forest, Gradient Boosting)
    """
    return {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=0.1, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            random_state=random_state
        ),
    }


def build_pipeline(preprocessor: ColumnTransformer, regressor: Any) -> Pipeline:
    """Combine preprocessor and regressor into a single leakage-free pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def run_grid_search_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    param_grid: Dict[str, Any] = None,
    cv: int = CV_FOLDS,
    n_jobs: int = -1,
) -> Tuple[GridSearchCV, Pipeline]:
    """
    Execute GridSearchCV for Random Forest regressor pipeline.
    Optimizes hyperparameters using cross-validation.
    """
    if param_grid is None:
        param_grid = {
            "regressor__n_estimators": [50, 100, 150],
            "regressor__max_depth": [None, 10, 20],
            "regressor__min_samples_split": [2, 5],
            "regressor__min_samples_leaf": [1, 2],
            "regressor__max_features": ["sqrt", 1.0],
        }

    base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
    pipeline = build_pipeline(preprocessor, base_rf)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    print(f"Starting Random Forest Grid Search ({len(grid_search.param_grid)} parameter combinations, {cv} folds)...")
    grid_search.fit(X_train, y_train)
    print("Grid search completed.")
    print(f"Best RMSE (CV): {-grid_search.best_score_:.4f}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    return grid_search, grid_search.best_estimator_


def save_model(model: Any, filepath: Path = MODEL_SAVE_FILE, compress: int = 3) -> Path:
    """Save trained model or pipeline to disk using joblib with compression."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath, compress=compress)
    print(f"Model saved to {filepath}")
    return filepath


def load_model(filepath: Path = MODEL_SAVE_FILE) -> Any:
    """Load model or pipeline from disk using joblib."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)
