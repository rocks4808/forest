"""CLI Inference and Prediction Tool for CanopyIntel.

Usage:
    python -m src.predict --region reg_01
    python -m src.predict --top 10
    python -m src.predict --all --json
"""

import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

from src.config import ROOT_DIR, load_config
from src.data_loader import load_and_validate
from src.features import build_features


def parse_args():
    parser = argparse.ArgumentParser(description="CanopyIntel CLI Inference Tool")
    parser.add_argument("--region", type=str, default=None, help="Region ID to predict (e.g. reg_01)")
    parser.add_argument("--top", type=int, default=None, help="Show top N highest risk regions")
    parser.add_argument("--all", action="store_true", help="Predict for all 50 regions")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    RAW_DATA_PATH = os.path.join(ROOT_DIR, cfg.get("data", {}).get("raw_path", "forest_loss_trend.csv"))

    reg_model_path = os.path.join(MODELS_DIR, "final_regression_model.joblib")
    clf_model_path = os.path.join(MODELS_DIR, "final_classification_model.joblib")
    feat_cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    thresholds_path = os.path.join(MODELS_DIR, "thresholds.json")

    if not os.path.exists(reg_model_path):
        print(f"Error: Trained model not found at {reg_model_path}. Run 'python -m src.train' first.")
        sys.exit(1)

    reg_model = joblib.load(reg_model_path)
    clf_model = joblib.load(clf_model_path) if os.path.exists(clf_model_path) else None

    with open(feat_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Load data and build features for 2024 (predicting 2025)
    data_file = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else os.path.join(ROOT_DIR, "forest_loss_trend.csv")
    df_raw = load_and_validate(data_file)
    training_stats = joblib.load(os.path.join(MODELS_DIR, "training_stats.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "training_stats.joblib")) else None
    df_feat, _ = build_features(df_raw, training_stats)

    latest_year = df_feat["year"].max()
    df_latest = df_feat[df_feat["year"] == latest_year].copy()

    if args.region:
        df_target = df_latest[df_latest["region_id"] == args.region].copy()
        if len(df_target) == 0:
            print(f"Error: Region '{args.region}' not found. Available: R01 to R50.")
            sys.exit(1)
    else:
        df_target = df_latest.copy()

    X = df_target[feature_cols].copy()

    # Predict residual log-ratio
    z_pred = reg_model.predict(X)
    y_current = df_target["loss_pct"].values
    y_pred = np.clip(y_current * np.exp(z_pred), 0.0, 100.0)

    # Forest area and projected loss ha
    area_current = df_target["forest_area_ha"].values
    loss_ha_pred = area_current * (y_pred / 100.0)

    # Classification
    tier_map = {0: "Low", 1: "Medium", 2: "High", 0.0: "Low", 1.0: "Medium", 2.0: "High"}
    if clf_model is not None:
        raw_preds = clf_model.predict(X)
        tiers = [tier_map.get(p, str(p)) for p in raw_preds]
    else:
        t1 = thresholds.get("T1", 0.5097)
        t2 = thresholds.get("T2", 1.0543)
        tiers = np.where(y_pred < t1, "Low", np.where(y_pred < t2, "Medium", "High"))

    results = []
    for idx, (_, row) in enumerate(df_target.iterrows()):
        results.append({
            "region_id": row["region_id"],
            "base_year": int(latest_year),
            "target_year": int(latest_year + 1),
            "current_loss_pct": round(float(row["loss_pct"]), 4),
            "pred_loss_pct": round(float(y_pred[idx]), 4),
            "pred_loss_ha": round(float(loss_ha_pred[idx]), 1),
            "severity_tier": str(tiers[idx]),
            "policy_active": bool(row["policy_active"]),
        })

    # Sort descending by predicted loss percentage
    results = sorted(results, key=lambda x: x["pred_loss_pct"], reverse=True)

    if args.top:
        results = results[:args.top]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Formatted terminal table
    print("\n" + "=" * 80)
    print(f" CANOPYINTEL FORECAST ENGINE | Target Year: {latest_year + 1}")
    print("=" * 80)
    print(f"{'Region':<10} {'Current Loss%':<15} {'Pred Loss%':<14} {'Pred Loss (ha)':<16} {'Severity':<10} {'Policy':<8}")
    print("-" * 80)
    for r in results:
        policy_str = "YES" if r["policy_active"] else "NO"
        print(f"{r['region_id']:<10} {r['current_loss_pct']:<15.4f} {r['pred_loss_pct']:<14.4f} {r['pred_loss_ha']:<16.1f} {r['severity_tier']:<10} {policy_str:<8}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
