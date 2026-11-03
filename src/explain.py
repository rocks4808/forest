"""Explainability module for the Forest Loss Prediction pipeline.

Provides:
- SHAP values via TreeSHAP (summary, bar, and waterfall plots).
- Permutation feature importance on validation data.
- Top drivers extraction per region/observation.
- Plain-language attribution sentence generation for dashboard.
- Partial dependence computation.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

REPORTS_FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "reports" / "figures"


# ===================================================================
# TreeSHAP Computation
# ===================================================================

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Compute TreeSHAP values for tree-based models (XGBoost, LightGBM, RF).

    Returns (explainer, shap_values_array).
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):  # Multiclass or list output
            shap_values = np.array(shap_values)
        return explainer, shap_values
    except Exception as e:
        logger.warning("TreeSHAP calculation failed or model not tree-based: %s", e)
        # Fallback to general Explainer with small background
        try:
            import shap
            sample_bg = X.iloc[: min(50, len(X))]
            explainer = shap.Explainer(model.predict, sample_bg)
            shap_exp = explainer(X)
            return explainer, shap_exp.values
        except Exception as e2:
            logger.error("General SHAP explainer also failed: %s", e2)
            return None, None


# ===================================================================
# SHAP Visualizations
# ===================================================================

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> str:
    """Generate and save SHAP beeswarm / summary plot."""
    import shap

    save_path = save_path or str(REPORTS_FIGURES_DIR / "shap_summary.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_shap_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> str:
    """Generate and save SHAP feature importance bar plot (mean |SHAP|)."""
    import shap

    save_path = save_path or str(REPORTS_FIGURES_DIR / "shap_bar.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_shap_waterfall(
    explainer: Any,
    shap_values_row: np.ndarray,
    X_row: pd.Series,
    save_path: Optional[str] = None,
    max_display: int = 10,
) -> str:
    """Generate and save a local waterfall attribution plot for a single instance."""
    import shap

    save_path = save_path or str(REPORTS_FIGURES_DIR / "shap_waterfall.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
    except Exception:
        base_val = 0.0

    # Build Explanation object for row
    exp = shap.Explanation(
        values=shap_values_row,
        base_values=base_val,
        data=X_row.values,
        feature_names=list(X_row.index),
    )

    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ===================================================================
# Permutation Feature Importance
# ===================================================================

def compute_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compute permutation importance on the holdout validation set."""
    r = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    perm_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": -r.importances_mean,  # flip sign so higher = more impactful
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        perm_df.to_csv(save_path, index=False)

    return perm_df


# ===================================================================
# Plain-Language Attribution Sentences & Top Drivers
# ===================================================================

def extract_top_drivers(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 3,
) -> List[List[Dict[str, Any]]]:
    """Extract top positive and negative driver names and values for each row."""
    results = []
    for i in range(len(shap_values)):
        row_vals = shap_values[i]
        top_idx = np.argsort(np.abs(row_vals))[::-1][:top_n]
        drivers = [
            {
                "feature": feature_names[idx],
                "shap_value": float(row_vals[idx]),
                "direction": "increases_loss" if row_vals[idx] > 0 else "reduces_loss",
            }
            for idx in top_idx
        ]
        results.append(drivers)
    return results


def format_top_drivers_string(drivers: List[Dict[str, Any]]) -> str:
    """Format drivers as comma-separated string for CSV export."""
    items = []
    for d in drivers:
        sign = "+" if d["shap_value"] > 0 else "-"
        items.append(f"{d['feature']} ({sign}{abs(d['shap_value']):.3f})")
    return "; ".join(items)


def generate_shap_sentence(
    shap_values_row: np.ndarray,
    feature_names: List[str],
    prediction: float,
    baseline_val: Optional[float] = None,
    top_n: int = 3,
) -> str:
    """Generate an editorial plain-language sentence explaining a prediction."""
    top_idx = np.argsort(np.abs(shap_values_row))[::-1][:top_n]
    
    pos_drivers = []
    neg_drivers = []
    for idx in top_idx:
        val = shap_values_row[idx]
        name = feature_names[idx].replace("_", " ").title()
        if val > 0:
            pos_drivers.append(f"{name} (+{val:.2f}%)")
        else:
            neg_drivers.append(f"{name} ({val:.2f}%)")

    pred_str = f"Predicted loss is {prediction:.2f}%."
    reasons = []
    if pos_drivers:
        reasons.append(f"driven upward by {', '.join(pos_drivers)}")
    if neg_drivers:
        reasons.append(f"dampened by {', '.join(neg_drivers)}")

    if reasons:
        return f"{pred_str} Primary factors: {'; '.join(reasons)}."
    return f"{pred_str} Influenced by regional baseline trends."

