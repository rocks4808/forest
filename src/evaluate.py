"""Evaluation metrics module for the Forest Loss Prediction pipeline.

Covers:
- Regression metrics: MAE, RMSE, R², sMAPE, MASE, per-region/year MAE.
- Classification metrics: Accuracy, Macro/Weighted F1, Macro Precision/Recall,
  MCC, Cohen's kappa, ROC-AUC (OvR), adjacent-class accuracy, Confusion Matrix.
- Bootstrap confidence intervals for headline numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Regression Metrics
# ===================================================================

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute symmetric Mean Absolute Percentage Error (sMAPE).

    Formula: (100 / n) * sum(2 * |y - ŷ| / (|y| + |ŷ| + eps))
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(y_t) + np.abs(y_p)) / 2.0
    # Avoid zero division
    zero_mask = (y_t == 0) & (y_p == 0)
    denom_safe = np.where(zero_mask, 1.0, denominator)
    diff = np.abs(y_p - y_t) / denom_safe
    diff[zero_mask] = 0.0
    return float(np.mean(diff) * 100.0)


def calculate_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: Optional[np.ndarray] = None,
) -> float:
    """Compute Mean Absolute Scaled Error relative to persistence baseline.

    If y_persistence is provided, MASE = MAE(y_true, y_pred) / MAE(y_true, y_persistence).
    Value < 1 means the model outperforms persistence.
    """
    mae_model = float(mean_absolute_error(y_true, y_pred))
    if y_persistence is not None:
        mae_base = float(mean_absolute_error(y_true, y_persistence))
        if mae_base == 0:
            return 1.0 if mae_model == 0 else float("inf")
        return mae_model / mae_base
    return float("nan")


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_persistence: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute all standard regression metrics for forest loss predictions."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    
    # Mask out NaNs
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    if y_persistence is not None:
        y_pers = np.asarray(y_persistence, dtype=np.float64)
        valid = valid & ~np.isnan(y_pers)
    else:
        y_pers = None

    y_t = y_t[valid]
    y_p = y_p[valid]
    if y_pers is not None:
        y_pers = y_pers[valid]

    if len(y_t) == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "sMAPE": float("nan"),
            "MASE": float("nan"),
        }

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))
    smape = calculate_smape(y_t, y_p)
    mase = calculate_mase(y_t, y_p, y_pers)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE": smape,
        "MASE": mase,
    }


def compute_error_breakdown(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str = "region_id",
) -> pd.DataFrame:
    """Compute per-group (e.g. region_id or year) MAE, RMSE, and bias."""
    records = []
    for grp, sub in df.groupby(group_col):
        yt = sub[y_true_col].dropna().values
        yp = sub[y_pred_col].dropna().values
        if len(yt) > 0 and len(yt) == len(yp):
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            bias = float(np.mean(yp - yt))
            records.append({group_col: grp, "MAE": mae, "RMSE": rmse, "bias": bias, "count": len(yt)})
    return pd.DataFrame(records)


# ===================================================================
# Classification Metrics
# ===================================================================

def calculate_adjacent_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute adjacent-class accuracy: fraction of predictions within |true - pred| <= 1.

    Particularly meaningful for ordinal classes (Low=0, Medium=1, High=2).
    """
    diff = np.abs(np.asarray(y_true, dtype=np.int64) - np.asarray(y_pred, dtype=np.int64))
    return float(np.mean(diff <= 1))


def calculate_precision_at_k(
    y_true_high: np.ndarray,
    risk_scores: np.ndarray,
    k: int = 10,
) -> float:
    """Compute precision@K: of the top-K highest risk regions, what fraction are truly High severity."""
    if len(risk_scores) < k:
        k = len(risk_scores)
    top_k_idx = np.argsort(risk_scores)[::-1][:k]
    return float(np.mean(y_true_high[top_k_idx]))


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Metrics:
    - Accuracy
    - Macro-Precision, Macro-Recall, Macro-F1
    - Weighted-F1
    - Matthews Correlation Coefficient (MCC)
    - Cohen's kappa (quadratic weighted for ordinal severity)
    - ROC-AUC (One-vs-Rest, Macro) if probabilities provided
    - Adjacent-class accuracy
    - Confusion matrix (both raw counts and normalized)
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    if labels is None:
        labels = sorted(list(np.unique(np.concatenate([y_t, y_p]))))

    acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=labels, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p))
    
    # Quadratic weighted kappa for ordinal classification
    try:
        kappa = float(cohen_kappa_score(y_t, y_p, weights="quadratic"))
    except Exception:
        kappa = float(cohen_kappa_score(y_t, y_p))

    adj_acc = calculate_adjacent_class_accuracy(y_t, y_p)

    cm = confusion_matrix(y_t, y_p, labels=labels)
    cm_norm = confusion_matrix(y_t, y_p, labels=labels, normalize="true")

    # ROC-AUC if probabilities available
    roc_auc = float("nan")
    if y_proba is not None:
        try:
            if len(labels) == 2:
                # Binary
                prob_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                roc_auc = float(roc_auc_score(y_t, prob_pos))
            else:
                # Multiclass OvR
                roc_auc = float(roc_auc_score(y_t, y_proba, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.debug("ROC-AUC computation failed: %s", e)
            roc_auc = float("nan")

    return {
        "Accuracy": acc,
        "Macro_Precision": macro_prec,
        "Macro_Recall": macro_rec,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "MCC": mcc,
        "Cohen_Kappa": kappa,
        "Adjacent_Accuracy": adj_acc,
        "ROC_AUC": roc_auc,
        "Confusion_Matrix": cm.tolist(),
        "Confusion_Matrix_Normalized": cm_norm.tolist(),
        "Labels": [int(l) for l in labels],
    }


# ===================================================================
# Bootstrap Confidence Intervals
# ===================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval for a metric.

    Returns:
    (point_estimate, lower_bound, upper_bound)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    n = len(y_t)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point_est = float(metric_fn(y_t, y_p))
    rng = np.random.RandomState(seed)
    boot_estimates = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_fn(y_t[idx], y_p[idx])
            if not np.isnan(val):
                boot_estimates.append(val)
        except Exception:
            continue

    if len(boot_estimates) < 20:
        return point_est, point_est, point_est

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_estimates, 100.0 * alpha))
    hi = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha)))
    return point_est, lo, hi


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = mean_absolute_error,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap comparison between Model A and Model B.

    Difference = Metric(Model A) - Metric(Model B).
    For MAE, a negative difference indicates Model A is better.
    """
    y_t = np.asarray(y_true)
    ya = np.asarray(y_pred_a)
    yb = np.asarray(y_pred_b)
    n = len(y_t)

    score_a = float(metric_fn(y_t, ya))
    score_b = float(metric_fn(y_t, yb))
    diff_point = score_a - score_b

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            d = metric_fn(y_t[idx], ya[idx]) - metric_fn(y_t[idx], yb[idx])
            diffs.append(d)
        except Exception:
            continue

    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.percentile(diffs, 100.0 * alpha))
    ci_hi = float(np.percentile(diffs, 100.0 * (1.0 - alpha)))
    p_value = float(np.mean(np.array(diffs) >= 0) if diff_point < 0 else np.mean(np.array(diffs) <= 0))

    return {
        "score_a": score_a,
        "score_b": score_b,
        "diff": diff_point,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": p_value,
    }

