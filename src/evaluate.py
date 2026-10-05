"""
Evaluation, metrics computation, cross-validation, and visualization module.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline

from src.config import ARTIFACTS_DIR, RANDOM_STATE, CV_FOLDS


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate RMSE, MAE, R-squared, and MAPE metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE (%)": float(mape),
    }


def evaluate_model_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Perform k-fold cross-validation and return mean & std of metrics."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    scores = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=-1)

    return {
        "CV_RMSE_mean": float(-scores["test_rmse"].mean()),
        "CV_RMSE_std": float(scores["test_rmse"].std()),
        "CV_MAE_mean": float(-scores["test_mae"].mean()),
        "CV_MAE_std": float(scores["test_mae"].std()),
        "CV_R2_mean": float(scores["test_r2"].mean()),
        "CV_R2_std": float(scores["test_r2"].std()),
    }


def plot_eda_distributions(df: pd.DataFrame, output_path: Path = None) -> Path:
    """Plot distribution histograms and boxplots for price and parameters."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "eda_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exploratory Data Analysis: Feature & Target Distributions", fontsize=16, fontweight="bold")

    # 1. Price histogram (raw vs log)
    sns.histplot(df["price"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=40)
    axes[0, 0].set_title("Price Distribution (Target)")
    axes[0, 0].set_xlabel("Price ($)")

    # 2. Price by Day of Week
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    existing_dow = [d for d in dow_order if d in df["dow"].unique()]
    sns.boxplot(
        data=df,
        x="dow",
        y="price",
        hue="dow",
        order=existing_dow,
        ax=axes[0, 1],
        palette="Blues_d",
        legend=False,
    )
    axes[0, 1].set_title("Price by Day of Week")
    axes[0, 1].set_xlabel("Day of Week")
    axes[0, 1].set_ylabel("Price ($)")

    # 3. para1 distribution
    sns.histplot(df["para1"], kde=False, discrete=True, ax=axes[0, 2], color="#2ca02c")
    axes[0, 2].set_title("Parameter 1 (Discrete Values)")
    axes[0, 2].set_xlabel("para1")

    # 4. para2 distribution
    sns.histplot(df["para2"], kde=True, ax=axes[1, 0], color="#ff7f0e", bins=30)
    axes[1, 0].set_title("Parameter 2 Distribution")
    axes[1, 0].set_xlabel("para2")

    # 5. para3 distribution
    sns.histplot(df["para3"], kde=True, ax=axes[1, 1], color="#9467bd", bins=30)
    axes[1, 1].set_title("Parameter 3 Distribution")
    axes[1, 1].set_xlabel("para3")

    # 6. para4 distribution
    sns.histplot(df["para4"], kde=True, ax=axes[1, 2], color="#d62728", bins=30)
    axes[1, 2].set_title("Parameter 4 Distribution")
    axes[1, 2].set_xlabel("para4")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path = None) -> Path:
    """Plot correlation heatmap of numeric features and price."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "correlation_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
    )
    plt.title("Correlation Matrix (Numeric Features & Target)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_model_comparison(results_df: pd.DataFrame, output_path: Path = None) -> Path:
    """Plot benchmark comparison of candidate models across RMSE, MAE, and R2."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Candidate Model Performance Comparison", fontsize=15, fontweight="bold")

    # Sort models by Test R2 descending
    sorted_df = results_df.sort_values(by="Test_R2", ascending=False)

    # 1. R-Squared
    sns.barplot(data=sorted_df, x="Test_R2", y="Model", hue="Model", ax=axes[0], palette="viridis", legend=False)
    axes[0].set_title("Test R² Score (Higher is Better)")
    axes[0].set_xlabel("R²")
    axes[0].set_xlim(-0.05, 1.0)

    # 2. RMSE
    sns.barplot(data=sorted_df, x="Test_RMSE", y="Model", hue="Model", ax=axes[1], palette="magma", legend=False)
    axes[1].set_title("Test RMSE (Lower is Better)")
    axes[1].set_xlabel("RMSE ($)")

    # 3. MAE
    sns.barplot(data=sorted_df, x="Test_MAE", y="Model", hue="Model", ax=axes[2], palette="crest", legend=False)
    axes[2].set_title("Test MAE (Lower is Better)")
    axes[2].set_xlabel("MAE ($)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
    output_path: Path = None,
) -> Path:
    """Generate diagnostic residual plots: Actual vs Predicted and Residual distribution."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "residuals_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Residual & Diagnostic Analysis: {model_name}", fontsize=15, fontweight="bold")

    # 1. Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.4, color="#1f77b4", edgecolor="none", s=25)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    axes[0].set_title("Actual vs. Predicted Price")
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Residuals vs Predicted (Homoscedasticity)
    axes[1].scatter(y_pred, residuals, alpha=0.4, color="#ff7f0e", edgecolor="none", s=25)
    axes[1].axhline(0, color="r", linestyle="--", lw=2)
    axes[1].set_title("Residuals vs. Predicted Price")
    axes[1].set_xlabel("Predicted Price ($)")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].grid(True, alpha=0.3)

    # 3. Distribution of Residuals
    sns.histplot(residuals, kde=True, ax=axes[2], color="#2ca02c", bins=40)
    axes[2].axvline(0, color="r", linestyle="--", lw=2)
    axes[2].set_title("Residual Distribution (Error Normality)")
    axes[2].set_xlabel("Residual ($)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_importance(
    rf_model: Any,
    feature_names: List[str],
    output_path: Path = None,
) -> Path:
    """Plot feature importance ranking from trained Random Forest regressor."""
    if output_path is None:
        output_path = ARTIFACTS_DIR / "feature_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
    elif hasattr(rf_model, "named_steps") and hasattr(rf_model.named_steps["regressor"], "feature_importances_"):
        importances = rf_model.named_steps["regressor"].feature_importances_
    else:
        raise ValueError("Model does not provide feature_importances_ attribute.")

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
    plt.title("Random Forest: Feature Importance Ranking", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Gini / Impurity Reduction Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
