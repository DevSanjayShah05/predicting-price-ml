#!/usr/bin/env python3
"""
End-to-end execution pipeline for Haensel AMS Price Prediction project.
Runs data loading, cleaning, EDA visualization, multi-model evaluation,
Random Forest Grid Search optimization, residual analysis, and model serialization.
"""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    ARTIFACTS_DIR,
    MODEL_SAVE_FILE,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data_loader import load_raw_data
from src.preprocessing import (
    clean_raw_data,
    split_data,
    build_preprocessor,
    save_processed_data,
)
from src.models import (
    get_candidate_models,
    build_pipeline,
    run_grid_search_rf,
    save_model,
)
from src.evaluate import (
    compute_metrics,
    evaluate_model_cv,
    plot_eda_distributions,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_residuals,
    plot_feature_importance,
)


def main():
    print("=" * 70)
    print(" HAENSEL AMS: PREDICTING PRICE MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Load Raw Data
    print("\n[Step 1/8] Loading raw dataset...")
    df_raw = load_raw_data(RAW_DATA_FILE)
    print(f"Raw dataset shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    # 2. Clean Data & Audit
    print("\n[Step 2/8] Cleaning data and removing invalid entries/outliers...")
    df_clean = clean_raw_data(df_raw, filter_para1_outliers=True)
    rows_removed = len(df_raw) - len(df_clean)
    print(f"Cleaned dataset shape: {df_clean.shape[0]:,} rows ({rows_removed} removed)")
    save_processed_data(df_clean, CLEANED_DATA_FILE)
    print(f"Cleaned data saved to: {CLEANED_DATA_FILE}")

    # 3. Generate EDA Visualizations
    print("\n[Step 3/8] Generating EDA and correlation visualizations...")
    dist_plot = plot_eda_distributions(df_clean, ARTIFACTS_DIR / "eda_distributions.png")
    corr_plot = plot_correlation_heatmap(df_clean, ARTIFACTS_DIR / "correlation_heatmap.png")
    print(f"Saved distribution plots to: {dist_plot}")
    print(f"Saved correlation heatmap to: {corr_plot}")

    # 4. Train/Test Split (Strict Featurization Ordering)
    print("\n[Step 4/8] Splitting dataset into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = split_data(df_clean, random_state=RANDOM_STATE)
    print(f"Training features: {X_train.shape}, Test features: {X_test.shape}")

    # 5. Benchmark Data Scalers
    print("\n[Step 5/8] Comparing scaling techniques (Standard vs Min-Max vs Robust)...")
    from sklearn.linear_model import Ridge
    scaler_benchmarks = []
    for scaler_name in ["standard", "minmax", "robust"]:
        prep = build_preprocessor(scaling_method=scaler_name)
        pipe = build_pipeline(prep, Ridge(alpha=1.0, random_state=RANDOM_STATE))
        cv_res = evaluate_model_cv(pipe, X_train, y_train, cv=CV_FOLDS)
        scaler_benchmarks.append(
            {
                "Scaler": scaler_name.capitalize(),
                "CV_RMSE": cv_res["CV_RMSE_mean"],
                "CV_MAE": cv_res["CV_MAE_mean"],
                "CV_R2": cv_res["CV_R2_mean"],
            }
        )
    scaler_df = pd.DataFrame(scaler_benchmarks)
    print("\nScaler Comparison on Ridge Regression (5-Fold CV):")
    print(scaler_df.to_string(index=False))

    # 6. Fit & Evaluate Candidate Models
    print("\n[Step 6/8] Fitting and evaluating candidate models (5-Fold CV & Test Set)...")
    default_preprocessor = build_preprocessor(scaling_method="standard")
    candidate_models = get_candidate_models(random_state=RANDOM_STATE)

    model_results = []
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        start_t = time.time()
        pipeline = build_pipeline(default_preprocessor, model)

        # Cross-validation on training data
        cv_scores = evaluate_model_cv(pipeline, X_train, y_train, cv=CV_FOLDS)

        # Fit on full training set and evaluate on test set
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
        elapsed = time.time() - start_t

        fitted_pipelines[name] = pipeline
        model_results.append(
            {
                "Model": name,
                "CV_RMSE": cv_scores["CV_RMSE_mean"],
                "CV_R2": cv_scores["CV_R2_mean"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_MAE": test_metrics["MAE"],
                "Test_R2": test_metrics["R2"],
                "Test_MAPE(%)": test_metrics["MAPE (%)"],
                "Fit_Time(s)": round(elapsed, 2),
            }
        )

    results_df = pd.DataFrame(model_results)
    print("\nCandidate Model Performance Summary:")
    print(results_df.to_string(index=False))

    # Plot model comparison
    comp_plot = plot_model_comparison(results_df, ARTIFACTS_DIR / "model_comparison.png")
    print(f"Saved model comparison plot to: {comp_plot}")

    # 7. Hyperparameter Tuning: Random Forest Grid Search
    print("\n[Step 7/8] Running GridSearchCV on Random Forest...")
    rf_grid = {
        "regressor__n_estimators": [50, 100, 150],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
        "regressor__max_features": ["sqrt", 1.0],
    }
    grid_search, best_pipeline = run_grid_search_rf(
        X_train, y_train, default_preprocessor, param_grid=rf_grid, cv=CV_FOLDS
    )

    # Evaluate Best Model on Test Set
    y_test_best = best_pipeline.predict(X_test)
    best_metrics = compute_metrics(y_test, y_test_best)
    print("\nTuned Random Forest Test Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 8. Diagnostics & Serialization
    print("\n[Step 8/8] Generating residual diagnostics, feature importances, and saving model...")
    res_plot = plot_residuals(
        y_test, y_test_best, model_name="Tuned Random Forest", output_path=ARTIFACTS_DIR / "residuals_analysis.png"
    )
    print(f"Saved residual analysis to: {res_plot}")

    # Feature importances
    ohe_step = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_cats = ohe_step.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
    feature_names = NUMERICAL_COLUMNS + ohe_cats

    fi_plot = plot_feature_importance(
        best_pipeline, feature_names, output_path=ARTIFACTS_DIR / "feature_importance.png"
    )
    print(f"Saved feature importance plot to: {fi_plot}")

    # Save model artifact
    save_model(best_pipeline, MODEL_SAVE_FILE)

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" Artifacts directory: {ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
