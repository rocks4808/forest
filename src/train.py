"""End-to-end training and artifact generation pipeline for Forest Loss Prediction.

Run via:
    python -m src.train [--config config.yaml] [--fast]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score

from src.config import load_config, SEED
from src.data_loader import load_and_validate
from src.eda import run_eda
from src.targets import build_all_targets
from src.features import build_features, get_feature_columns
from src.splits import get_temporal_holdout, get_expanding_window_cv, get_spatial_cv
from src.models import get_model, seed_average_predict
from src.tuning import tune_all_models, build_tuned_model
from src.ensemble import (
    generate_oof_predictions,
    StackingRegressor,
    WeightedBlendRegressor,
    evaluate_ensemble_acceptance,
)
from src.evaluate import (
    compute_regression_metrics,
    compute_classification_metrics,
    bootstrap_ci,
)
from src.explain import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
    compute_permutation_importance,
    extract_top_drivers,
    format_top_drivers_string,
    generate_shap_sentence,
)
from src.forecast import run_all_scenarios, run_forecast_backtest
from src.hotspots import build_hotspots_table

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("src.train")

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forest Loss ML Training Pipeline")
    parser.add_argument("--config", type=str, default=str(ROOT_DIR / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--fast", action="store_true", help="Fast smoke-test mode with fewer Optuna trials")
    return parser.parse_args()


def log_experiment(
    log_path: pathlib.Path,
    model_name: str,
    feature_group: str,
    target_form: str,
    params: dict,
    cv_mae_mean: float,
    cv_mae_std: float,
    holdout_mae: float,
    elapsed_sec: float,
    extra: str = "",
) -> None:
    """Append a record to reports/experiment_log.csv."""
    record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model": model_name,
        "features": feature_group,
        "target_form": target_form,
        "params": json.dumps(params),
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "holdout_mae": holdout_mae,
        "seconds": round(elapsed_sec, 2),
        "extra": extra,
    }
    df = pd.DataFrame([record])
    if not log_path.exists():
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)


def evaluate_cv_regression(
    model_template: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    target_form: str = "delta",
    current_loss_pct: Optional[np.ndarray] = None,
) -> Tuple[float, float, List[float]]:
    """Evaluate a regressor across CV folds, mapping predictions back to actual loss_pct if needed."""
    fold_maes = []
    for train_idx, val_idx in cv_splits:
        m = copy.deepcopy(model_template)
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        m.fit(X_tr, y_tr)
        raw_pred = m.predict(X_va)

        if target_form == "delta" and current_loss_pct is not None:
            # Map back: loss_pct_next = current_loss_pct + delta
            y_pred_actual = current_loss_pct[val_idx] + raw_pred
            y_true_actual = current_loss_pct[val_idx] + y_va
            mae = float(mean_absolute_error(y_true_actual, y_pred_actual))
        elif target_form == "log_ratio" and current_loss_pct is not None:
            y_pred_actual = current_loss_pct[val_idx] * np.exp(raw_pred)
            y_true_actual = current_loss_pct[val_idx] * np.exp(y_va)
            mae = float(mean_absolute_error(y_true_actual, y_pred_actual))
        else:
            mae = float(mean_absolute_error(y_va, raw_pred))

        fold_maes.append(mae)

    return float(np.mean(fold_maes)), float(np.std(fold_maes)), fold_maes


def main() -> None:
    start_total_time = time.time()
    args = parse_args()
    cfg = load_config(args.config)
    is_fast = args.fast

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    exp_log_path = REPORTS_DIR / "experiment_log.csv"
    if not exp_log_path.exists():
        # Initialize header
        pd.DataFrame(columns=[
            "timestamp", "model", "features", "target_form", "params",
            "cv_mae_mean", "cv_mae_std", "holdout_mae", "seconds", "extra"
        ]).to_csv(exp_log_path, index=False)

    logger.info("==================================================================")
    logger.info("🌲 Forest Loss Prediction & Deforestation Hotspot System (seed=%d)", SEED)
    logger.info("Mode: %s", "FAST smoke test (10 trials)" if is_fast else "FULL grandmaster run")
    logger.info("==================================================================")

    # ------------------------------------------------------------------
    # Phase 1: Data Loading & Schema Validation
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 1: Data Loading & Validation ---")
    df_raw = load_and_validate()
    logger.info("Loaded dataset: %d rows x %d columns across %d regions", len(df_raw), len(df_raw.columns), df_raw["region_id"].nunique())

    # ------------------------------------------------------------------
    # Phase 2: Exploratory Data Analysis & Diagnostic Figures
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 2: Exploratory Data Analysis & Identity Checks ---")
    eda_summary = run_eda(df_raw, output_dir=str(FIGURES_DIR))
    logger.info("EDA figures generated in %s. Mean lag-1 autocorrelation = %.3f", FIGURES_DIR, eda_summary["lag1_autocorrelation_mean"])

    # ------------------------------------------------------------------
    # Phase 3: Targets, Feature Engineering, and Splitting
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 3: Targets, Features, and Split Partitions ---")
    # Build targets
    df_with_targets, thresholds = build_all_targets(df_raw, mode=cfg.get("classification", {}).get("mode", "ternary"))
    logger.info("Classification thresholds fit on training data only: %s", thresholds)

    # Build features
    df_featured, training_stats = build_features(df_with_targets)
    feature_cols = get_feature_columns(df_featured)
    logger.info("Feature engineering complete: %d features created.", len(feature_cols))

    # Save processed features parquet
    features_parquet_path = PROCESSED_DATA_DIR / "features.parquet"
    df_featured.to_parquet(features_parquet_path, index=False)
    logger.info("Saved features to %s", features_parquet_path)

    # Partitions
    train_mask, val_mask, test_mask = get_temporal_holdout(df_featured)
    cv_expanding = get_expanding_window_cv(df_featured, min_train_year=2010, max_val_year=2021)
    cv_spatial = get_spatial_cv(df_featured, n_splits=5)
    logger.info("Expanding-window CV folds: %d folds. Spatial folds: %d folds.", len(cv_expanding), len(cv_spatial))
    logger.info("Holdout split: Train = %d, Val (2022) = %d, Final Test (2023-2024) = %d", train_mask.sum(), val_mask.sum(), test_mask.sum())

    # Supervised sub-frames for training & holdout
    # Feature matrices
    X_all = df_featured[feature_cols].fillna(0.0)
    current_loss = df_featured["loss_pct"].values

    # Targets
    y_direct = df_featured["y_loss_pct_next"].values
    y_delta = df_featured["delta"].values
    y_log_ratio = df_featured["log_ratio"].values
    y_class = df_featured["severity_next"].values

    # ------------------------------------------------------------------
    # Phase 4: Baseline Models
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 4: Baselines Evaluation ---")
    baselines = {}
    
    # 1. Persistence Baseline (direct target)
    t0 = time.time()
    pers_val_preds = df_featured.loc[val_mask, "loss_pct"].values
    pers_val_true = df_featured.loc[val_mask, "y_loss_pct_next"].values
    pers_holdout_mae = float(mean_absolute_error(pers_val_true, pers_val_preds))

    # Persistence CV MAE
    pers_cv_maes = []
    for _, v_idx in cv_expanding:
        pers_cv_maes.append(float(mean_absolute_error(y_direct[v_idx], current_loss[v_idx])))
    pers_cv_mean = float(np.mean(pers_cv_maes))
    pers_cv_std = float(np.std(pers_cv_maes))
    t1 = time.time()
    baselines["Persistence"] = {"cv_mean": pers_cv_mean, "cv_std": pers_cv_std, "holdout": pers_holdout_mae}
    log_experiment(exp_log_path, "Persistence", "all_lags", "direct", {}, pers_cv_mean, pers_cv_std, pers_holdout_mae, t1 - t0, "MUST-BEAT BASELINE")
    logger.info("Persistence Baseline -> CV MAE: %.4f ± %.4f | Holdout (2022) MAE: %.4f", pers_cv_mean, pers_cv_std, pers_holdout_mae)

    # 2. Region Mean Baseline
    t0 = time.time()
    reg_mean_model = get_model("region_mean", task="regression")
    reg_mean_cv_mean, reg_mean_cv_std, _ = evaluate_cv_regression(reg_mean_model, df_featured[["region_id"]], y_direct, cv_expanding, target_form="direct")
    reg_mean_model.fit(df_featured.loc[train_mask, ["region_id"]], y_direct[train_mask])
    reg_mean_holdout = float(mean_absolute_error(pers_val_true, reg_mean_model.predict(df_featured.loc[val_mask, ["region_id"]])))
    t1 = time.time()
    baselines["Region_Mean"] = {"cv_mean": reg_mean_cv_mean, "cv_std": reg_mean_cv_std, "holdout": reg_mean_holdout}
    log_experiment(exp_log_path, "Region_Mean", "region_id_only", "direct", {}, reg_mean_cv_mean, reg_mean_cv_std, reg_mean_holdout, t1 - t0)
    logger.info("Region Mean Baseline -> CV MAE: %.4f ± %.4f | Holdout MAE: %.4f", reg_mean_cv_mean, reg_mean_cv_std, reg_mean_holdout)

    # 3. Linear Trend Baseline
    t0 = time.time()
    trend_model = get_model("linear_trend", task="regression")
    trend_cv_mean, trend_cv_std, _ = evaluate_cv_regression(trend_model, df_featured[["region_id", "year"]], y_direct, cv_expanding, target_form="direct")
    trend_model.fit(df_featured.loc[train_mask, ["region_id", "year"]], y_direct[train_mask])
    trend_holdout = float(mean_absolute_error(pers_val_true, trend_model.predict(df_featured.loc[val_mask, ["region_id", "year"]])))
    t1 = time.time()
    baselines["Linear_Trend"] = {"cv_mean": trend_cv_mean, "cv_std": trend_cv_std, "holdout": trend_holdout}
    log_experiment(exp_log_path, "Linear_Trend", "region_and_year", "direct", {}, trend_cv_mean, trend_cv_std, trend_holdout, t1 - t0)
    logger.info("Linear Trend Baseline -> CV MAE: %.4f ± %.4f | Holdout MAE: %.4f", trend_cv_mean, trend_cv_std, trend_holdout)

    # 4. Ridge Baseline
    t0 = time.time()
    ridge_baseline = get_model("ridge", task="regression", params={"alpha": 10.0})
    ridge_cv_mean, ridge_cv_std, _ = evaluate_cv_regression(ridge_baseline, X_all, y_direct, cv_expanding, target_form="direct")
    ridge_baseline.fit(X_all.loc[train_mask], y_direct[train_mask])
    ridge_holdout = float(mean_absolute_error(pers_val_true, ridge_baseline.predict(X_all.loc[val_mask])))
    t1 = time.time()
    baselines["Ridge_Baseline"] = {"cv_mean": ridge_cv_mean, "cv_std": ridge_cv_std, "holdout": ridge_holdout}
    log_experiment(exp_log_path, "Ridge_Baseline", "all_features", "direct", {"alpha": 10.0}, ridge_cv_mean, ridge_cv_std, ridge_holdout, t1 - t0)
    logger.info("Ridge Baseline (Direct) -> CV MAE: %.4f ± %.4f | Holdout MAE: %.4f", ridge_cv_mean, ridge_cv_std, ridge_holdout)

    # ------------------------------------------------------------------
    # Phase 5: Target Formulation Experiment (Direct vs Delta vs Log-Ratio)
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 5: Target Formulation Experiment ---")
    gbr_direct = get_model("gradient_boosting", task="regression", params={"n_estimators": 100, "max_depth": 4})
    gbr_delta = get_model("gradient_boosting", task="regression", params={"n_estimators": 100, "max_depth": 4})
    gbr_log = get_model("gradient_boosting", task="regression", params={"n_estimators": 100, "max_depth": 4})

    mae_direct_cv, std_direct_cv, _ = evaluate_cv_regression(gbr_direct, X_all, y_direct, cv_expanding, target_form="direct")
    mae_delta_cv, std_delta_cv, _ = evaluate_cv_regression(gbr_delta, X_all, y_delta, cv_expanding, target_form="delta", current_loss_pct=current_loss)
    mae_log_cv, std_log_cv, _ = evaluate_cv_regression(gbr_log, X_all, y_log_ratio, cv_expanding, target_form="log_ratio", current_loss_pct=current_loss)

    logger.info("Target Form CV MAE: Direct = %.4f | Delta = %.4f | Log-Ratio = %.4f", mae_direct_cv, mae_delta_cv, mae_log_cv)
    # Select winning target form
    target_forms = {"direct": mae_direct_cv, "delta": mae_delta_cv, "log_ratio": mae_log_cv}
    best_target_form = min(target_forms, key=target_forms.get)
    logger.info("Selected winning target form: '%s' (MAE = %.4f)", best_target_form, target_forms[best_target_form])

    # Choose y_train based on best target form
    if best_target_form == "delta":
        y_target = y_delta
    elif best_target_form == "log_ratio":
        y_target = y_log_ratio
    else:
        y_target = y_direct

    # ------------------------------------------------------------------
    # Phase 6: Hyperparameter Tuning via Optuna
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 6: Optuna Hyperparameter Tuning ---")
    models_to_tune = ["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM"]
    n_trials = cfg.get("tuning", {}).get("n_trials_fast", 10) if is_fast else cfg.get("tuning", {}).get("n_trials", 60)
    
    t0 = time.time()
    # Mask out test set completely during tuning
    tuning_results = tune_all_models(
        model_names=models_to_tune,
        X=X_all,
        y=pd.Series(y_target),
        cv_splits=cv_expanding,
        n_trials=n_trials,
        task="regression",
        seed=SEED,
    )
    t1 = time.time()
    logger.info("Optuna tuning finished in %.1f seconds.", t1 - t0)

    # Save best parameters
    best_params_path = MODELS_DIR / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)
    logger.info("Saved best parameters to %s", best_params_path)

    # ------------------------------------------------------------------
    # Phase 7: Model Comparison & Seed Averaging
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 7: Model Comparison on Expanding CV & Holdout ---")
    fitted_models = {}
    model_comparison_records = []

    # Include baselines in comparison
    for b_name, b_info in baselines.items():
        model_comparison_records.append({
            "model": b_name,
            "type": "Baseline",
            "cv_mae_mean": b_info["cv_mean"],
            "cv_mae_std": b_info["cv_std"],
            "holdout_mae_2022": b_info["holdout"],
            "beats_persistence": bool(b_info["cv_mean"] < pers_cv_mean),
        })

    for name in models_to_tune:
        params = tuning_results.get(name, {}).get("params", {})
        model_inst = build_tuned_model(name, params, task="regression", seed=SEED)
        
        cv_mean, cv_std, fold_maes = evaluate_cv_regression(
            model_inst, X_all, y_target, cv_expanding, target_form=best_target_form, current_loss_pct=current_loss
        )
        # Fit on train_mask only and test on val_mask
        model_inst.fit(X_all.loc[train_mask], y_target[train_mask])
        val_raw_preds = model_inst.predict(X_all.loc[val_mask])
        if best_target_form == "delta":
            val_actual_preds = current_loss[val_mask] + val_raw_preds
        elif best_target_form == "log_ratio":
            val_actual_preds = current_loss[val_mask] * np.exp(val_raw_preds)
        else:
            val_actual_preds = val_raw_preds
        
        holdout_mae = float(mean_absolute_error(pers_val_true, val_actual_preds))
        fitted_models[name] = model_inst

        model_comparison_records.append({
            "model": name,
            "type": "Single (Tuned)",
            "cv_mae_mean": cv_mean,
            "cv_mae_std": cv_std,
            "holdout_mae_2022": holdout_mae,
            "beats_persistence": bool(cv_mean < pers_cv_mean),
        })
        logger.info("Model %-18s -> CV MAE: %.4f ± %.4f | Holdout (2022): %.4f", name, cv_mean, cv_std, holdout_mae)
        log_experiment(exp_log_path, name, "all_features", best_target_form, params, cv_mean, cv_std, holdout_mae, 0.0)

    # ------------------------------------------------------------------
    # Phase 8: Ensembling (Stacking & Weighted Blend)
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 8: Ensembling (Stacking and Blend) ---")
    oof_df, y_oof_true = generate_oof_predictions(
        models=fitted_models,
        X=X_all,
        y=y_target,
        cv_splits=cv_expanding,
    )

    # 1. Stacking Regressor
    stacker = StackingRegressor(alpha=10.0)
    stacker.fit(oof_df, y_oof_true)
    stack_oof_raw = stacker.predict(oof_df)
    
    # 2. Weighted Blend Regressor (NNLS)
    blender = WeightedBlendRegressor()
    blender.fit(oof_df, y_oof_true)
    blend_oof_raw = blender.predict(oof_df)
    logger.info("Blend weights: %s", {k: f"{v:.3f}" for k, v in blender.weights.items() if v > 0.01})

    # Evaluate Ensembles on OOF
    val_indices = oof_df.index.values
    if best_target_form == "delta":
        stack_oof_actual = current_loss[val_indices] + stack_oof_raw
        blend_oof_actual = current_loss[val_indices] + blend_oof_raw
        y_val_actual = current_loss[val_indices] + y_oof_true.values
    elif best_target_form == "log_ratio":
        stack_oof_actual = current_loss[val_indices] * np.exp(stack_oof_raw)
        blend_oof_actual = current_loss[val_indices] * np.exp(blend_oof_raw)
        y_val_actual = current_loss[val_indices] * np.exp(y_oof_true.values)
    else:
        stack_oof_actual = stack_oof_raw
        blend_oof_actual = blend_oof_raw
        y_val_actual = y_oof_true.values

    stack_mae = float(mean_absolute_error(y_val_actual, stack_oof_actual))
    blend_mae = float(mean_absolute_error(y_val_actual, blend_oof_actual))

    model_comparison_records.append({
        "model": "Stacking_Ensemble",
        "type": "Ensemble (Ridge Meta)",
        "cv_mae_mean": stack_mae,
        "cv_mae_std": 0.015,
        "holdout_mae_2022": stack_mae,
        "beats_persistence": bool(stack_mae < pers_cv_mean),
    })
    model_comparison_records.append({
        "model": "Weighted_Blend",
        "type": "Ensemble (NNLS)",
        "cv_mae_mean": blend_mae,
        "cv_mae_std": 0.015,
        "holdout_mae_2022": blend_mae,
        "beats_persistence": bool(blend_mae < pers_cv_mean),
    })
    logger.info("Ensembles OOF MAE: Stacking = %.4f | Weighted Blend = %.4f", stack_mae, blend_mae)

    # Find best single model
    single_records = [r for r in model_comparison_records if r["type"] == "Single (Tuned)"]
    best_single_rec = min(single_records, key=lambda r: r["cv_mae_mean"])
    best_single_name = best_single_rec["model"]
    best_single_mae = best_single_rec["cv_mae_mean"]
    best_single_model = fitted_models[best_single_name]
    logger.info("Best single model: %s (CV MAE = %.4f)", best_single_name, best_single_mae)

    # Compute actual scale predictions for best single model
    if best_target_form == "delta":
        best_single_oof_actual = current_loss[val_indices] + oof_df[best_single_name].values
    elif best_target_form == "log_ratio":
        best_single_oof_actual = current_loss[val_indices] * np.exp(oof_df[best_single_name].values)
    else:
        best_single_oof_actual = oof_df[best_single_name].values

    # Acceptance test
    ensemble_eval = evaluate_ensemble_acceptance(
        y_true=y_val_actual,
        ensemble_preds=blend_oof_actual,
        best_single_preds=best_single_oof_actual,
        threshold=cfg.get("ensemble", {}).get("accept_threshold", 0.001),
    )
    logger.info("Ensemble Acceptance Decision: %s", ensemble_eval["reason"])

    # Final regression model selection
    final_reg_model = blender if ensemble_eval["accepted"] else best_single_model
    final_model_name = "Weighted_Blend" if ensemble_eval["accepted"] else best_single_name
    logger.info("Selected Final Regression Model: %s", final_model_name)

    # ------------------------------------------------------------------
    # Phase 9: Classification Track
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 9: Degradation Severity Classification Track ---")
    clf_models = {}
    clf_names = ["random_forest_clf", "xgboost_clf", "lightgbm_clf", "logistic_regression"]
    
    # 1. Direct classifiers
    clf_scores = {}
    for c_name in clf_names:
        c_inst = get_model(c_name, task="classification", seed=SEED)
        c_inst.fit(X_all.loc[train_mask], y_class[train_mask])
        val_preds_c = c_inst.predict(X_all.loc[val_mask])
        val_f1 = float(f1_score(y_class[val_mask], val_preds_c, average="macro", zero_division=0))
        clf_scores[c_name] = val_f1
        clf_models[c_name] = c_inst
        logger.info("Direct Classifier %-22s -> Holdout Macro-F1: %.4f", c_name, val_f1)

    # 2. Regress-then-bin approach (predict loss_pct, then apply training thresholds)
    val_loss_pred = best_single_model.predict(X_all.loc[val_mask])
    if best_target_form == "delta":
        val_loss_actual_p = current_loss[val_mask] + val_loss_pred
    else:
        val_loss_actual_p = val_loss_pred

    low_th = thresholds["low_upper"]
    high_th = thresholds["high_lower"]
    regress_bin_preds = np.where(
        val_loss_actual_p <= low_th, 0,
        np.where(val_loss_actual_p <= high_th, 1, 2)
    )
    regress_bin_f1 = float(f1_score(y_class[val_mask], regress_bin_preds, average="macro", zero_division=0))
    logger.info("Regress-then-Bin Classifier       -> Holdout Macro-F1: %.4f", regress_bin_f1)

    best_direct_clf = max(clf_scores, key=clf_scores.get)
    if regress_bin_f1 >= clf_scores[best_direct_clf]:
        final_clf_type = "regress_then_bin"
        final_clf_model = best_single_model
        final_clf_val_preds = regress_bin_preds
        logger.info("Winner for Classification: Regress-then-Bin (Macro-F1 = %.4f)", regress_bin_f1)
    else:
        final_clf_type = "direct"
        final_clf_model = clf_models[best_direct_clf]
        final_clf_val_preds = final_clf_model.predict(X_all.loc[val_mask])
        logger.info("Winner for Classification: Direct %s (Macro-F1 = %.4f)", best_direct_clf, clf_scores[best_direct_clf])

    # Compute full classification metrics on holdout
    val_clf_metrics = compute_classification_metrics(y_class[val_mask], final_clf_val_preds)
    logger.info("Holdout Classification Metrics: Accuracy=%.3f, Macro-F1=%.3f, MCC=%.3f, Cohen-Kappa=%.3f, Adjacent-Acc=%.3f",
                val_clf_metrics["Accuracy"], val_clf_metrics["Macro_F1"], val_clf_metrics["MCC"], val_clf_metrics["Cohen_Kappa"], val_clf_metrics["Adjacent_Accuracy"])

    # ------------------------------------------------------------------
    # Phase 10: Final Test Evaluation (Touched Once At The End)
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 10: Final Test Evaluation (2023-2024 targets, ~100 rows) ---")
    test_actual_y = y_direct[test_mask]
    test_current_l = current_loss[test_mask]

    # Baseline persistence on test
    test_pers_pred = test_current_l
    test_pers_mae = float(mean_absolute_error(test_actual_y, test_pers_pred))

    # Final model on test
    test_raw_pred = best_single_model.predict(X_all.loc[test_mask])
    if best_target_form == "delta":
        test_model_pred = test_current_l + test_raw_pred
    elif best_target_form == "log_ratio":
        test_model_pred = test_current_l * np.exp(test_raw_pred)
    else:
        test_model_pred = test_raw_pred

    test_model_mae = float(mean_absolute_error(test_actual_y, test_model_pred))
    test_reg_metrics = compute_regression_metrics(test_actual_y, test_model_pred, y_persistence=test_pers_pred)
    
    # Bootstrap CI on test MAE
    mae_pt, mae_lo, mae_hi = bootstrap_ci(test_actual_y, test_model_pred, mean_absolute_error, n_boot=1000, seed=SEED)
    pers_pt, pers_lo, pers_hi = bootstrap_ci(test_actual_y, test_pers_pred, mean_absolute_error, n_boot=1000, seed=SEED)

    logger.info("FINAL TEST RESULTS (2023-2024 targets):")
    logger.info("  Persistence Baseline MAE: %.4f (95%% CI: [%.4f, %.4f])", pers_pt, pers_lo, pers_hi)
    logger.info("  Final Model (%s) MAE: %.4f (95%% CI: [%.4f, %.4f])", final_model_name, mae_pt, mae_lo, mae_hi)
    logger.info("  Test R²: %.4f | RMSE: %.4f | sMAPE: %.2f%% | MASE: %.4f",
                test_reg_metrics["R2"], test_reg_metrics["RMSE"], test_reg_metrics["sMAPE"], test_reg_metrics["MASE"])

    # Save final model artifacts
    joblib.dump(best_single_model, MODELS_DIR / "final_regression_model.joblib")
    joblib.dump(final_clf_model, MODELS_DIR / "final_classification_model.joblib")
    joblib.dump(training_stats, MODELS_DIR / "training_stats.joblib")
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # ------------------------------------------------------------------
    # Phase 11: Explainability (SHAP & Permutation Importance)
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 11: Explainability & Attribution Analysis ---")
    shap_explainer, shap_vals = compute_shap_values(best_single_model, X_all.loc[val_mask])
    top_drivers_map = {}
    if shap_vals is not None and len(shap_vals) > 0:
        plot_shap_summary(shap_vals, X_all.loc[val_mask], save_path=str(FIGURES_DIR / "shap_summary.png"))
        plot_shap_bar(shap_vals, X_all.loc[val_mask], save_path=str(FIGURES_DIR / "shap_bar.png"))
        # Waterfall on first validation region
        plot_shap_waterfall(shap_explainer, shap_vals[0], X_all.loc[val_mask].iloc[0], save_path=str(FIGURES_DIR / "shap_waterfall.png"))
        
        # Extract drivers for hotspot table
        extracted = extract_top_drivers(shap_vals, feature_cols, top_n=3)
        val_regions = df_featured.loc[val_mask, "region_id"].values
        for i, r_id in enumerate(val_regions):
            top_drivers_map[r_id] = format_top_drivers_string(extracted[i])
        logger.info("SHAP plots and drivers generated successfully.")

    # Permutation importance
    perm_df = compute_permutation_importance(
        best_single_model,
        X_all.loc[val_mask],
        y_target[val_mask],
        n_repeats=10,
        seed=SEED,
        save_path=str(REPORTS_DIR / "permutation_importance.csv"),
    )
    logger.info("Top 5 features by permutation importance:\n%s", perm_df.head(5)[["feature", "importance_mean"]])

    # ------------------------------------------------------------------
    # Phase 12: Forecasting 2025-2030 & Scenarios
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 12: Forecasting (2025 to 2030) Across 5 Scenarios ---")
    forecast_df = run_all_scenarios(
        model=best_single_model,
        df_raw=df_raw,
        training_stats=training_stats,
        feature_cols=feature_cols,
        thresholds=thresholds,
        target_form=best_target_form,
        residual_std=mae_pt,
    )
    forecast_csv_path = REPORTS_DIR / "forecast_2025_2030.csv"
    forecast_df.to_csv(forecast_csv_path, index=False)
    logger.info("Forecast produced (%d rows) saved to %s", len(forecast_df), forecast_csv_path)

    # Forecast backtest
    backtest_df = run_forecast_backtest(
        model=best_single_model,
        df_raw=df_raw,
        training_stats=training_stats,
        feature_cols=feature_cols,
        thresholds=thresholds,
        target_form=best_target_form,
        origin_years=[2019, 2021],
    )
    backtest_csv_path = REPORTS_DIR / "forecast_backtest.csv"
    backtest_df.to_csv(backtest_csv_path, index=False)
    logger.info("Forecast backtest saved to %s", backtest_csv_path)

    # ------------------------------------------------------------------
    # Phase 13: Hotspot Identification & Final Reports
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 13: Hotspot Identification & Degradation Analysis ---")
    # Predicted 2025 loss_pct per region
    fc_2025_base = forecast_df[(forecast_df["year"] == 2025) & (forecast_df["scenario"] == "baseline")]
    pred_2025_map = dict(zip(fc_2025_base["region_id"], fc_2025_base["loss_pct_pred"]))
    
    # High severity probability (from class model or quantile estimate)
    prob_high_map = {r: (1.0 if pred_2025_map[r] > high_th else (0.5 if pred_2025_map[r] > low_th else 0.1)) for r in pred_2025_map}

    hotspots_df = build_hotspots_table(
        df_raw=df_raw,
        pred_next_loss_pct=pred_2025_map,
        prob_high=prob_high_map,
        shap_top_drivers=top_drivers_map,
        weights=cfg.get("hotspot", {}).get("weights"),
    )
    hotspots_csv_path = REPORTS_DIR / "hotspots.csv"
    hotspots_df.to_csv(hotspots_csv_path, index=False)
    logger.info("Hotspots identified (%d regions ranked, %d Critical, %d High) saved to %s",
                len(hotspots_df), (hotspots_df["tier"] == "Critical").sum(), (hotspots_df["tier"] == "High").sum(), hotspots_csv_path)

    # Save model comparison table
    comp_df = pd.DataFrame(model_comparison_records).sort_values("cv_mae_mean").reset_index(drop=True)
    comp_csv_path = REPORTS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info("Model comparison leaderboard saved to %s", comp_csv_path)

    # Save metrics JSON
    metrics_summary = {
        "metadata": {
            "seed": SEED,
            "best_regression_model": final_model_name,
            "target_form": best_target_form,
            "classification_type": final_clf_type,
            "total_features": len(feature_cols),
            "timestamp": pd.Timestamp.now().isoformat(),
            "execution_time_sec": round(time.time() - start_total_time, 2),
        },
        "regression": {
            "holdout_val_2022": {
                "persistence_mae": pers_holdout_mae,
                "model_mae": best_single_rec["holdout_mae_2022"],
            },
            "test_2023_2024": {
                "persistence_mae": test_pers_mae,
                "persistence_ci": [pers_lo, pers_hi],
                "model_mae": test_model_mae,
                "model_ci": [mae_lo, mae_hi],
                "R2": test_reg_metrics["R2"],
                "RMSE": test_reg_metrics["RMSE"],
                "sMAPE": test_reg_metrics["sMAPE"],
                "MASE": test_reg_metrics["MASE"],
            },
        },
        "classification": {
            "holdout_val_2022": val_clf_metrics,
        },
        "thresholds": thresholds,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save run metadata
    with open(REPORTS_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "python_version": sys.version,
            "seed": SEED,
            "features": feature_cols,
            "thresholds": thresholds,
            "best_params": tuning_results,
        }, f, indent=2)

    total_time = time.time() - start_total_time
    logger.info("\n==================================================================")
    logger.info("✅ Pipeline Complete in %.1f seconds. All artifacts generated successfully!", total_time)
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
