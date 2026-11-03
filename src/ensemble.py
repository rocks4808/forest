"""Ensemble module for the Forest Loss Prediction pipeline.

Provides:
- Out-of-fold (OOF) prediction generation across expanding-window CV folds.
- Stacking ensemble with Ridge / non-negative Ridge meta-learner.
- Optimal weighted blend via Non-Negative Least Squares (scipy.optimize.nnls).
- Paired bootstrap test to decide whether to ship ensemble or best single model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.evaluate import paired_bootstrap_test

logger = logging.getLogger(__name__)


# ===================================================================
# Out-of-Fold Predictions
# ===================================================================

def generate_oof_predictions(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    fit_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate out-of-fold (OOF) predictions for a dictionary of candidate models.

    Parameters
    ----------
    models : dict[str, BaseEstimator]
        Unfitted model templates (or cloned per fold).
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series or np.ndarray
        Full target vector.
    cv_splits : list of (train_idx, val_idx)
        Expanding-window CV splits.

    Returns
    -------
    oof_df : pd.DataFrame
        OOF predictions aligned to the validation rows across all folds.
    y_val_series : pd.Series
        Corresponding true target values.
    """
    import copy

    y_arr = np.asarray(y)
    model_names = list(models.keys())
    
    # Track indices that were validated
    val_indices = []
    for _, v_idx in cv_splits:
        val_indices.extend(v_idx)
    val_indices = np.array(val_indices)

    oof_preds = {name: np.full(len(X), np.nan) for name in model_names}

    for fold_i, (t_idx, v_idx) in enumerate(cv_splits):
        X_train, y_train = X.iloc[t_idx], y_arr[t_idx]
        X_val = X.iloc[v_idx]

        for name, model_template in models.items():
            model = copy.deepcopy(model_template)
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                oof_preds[name][v_idx] = preds
            except Exception as e:
                logger.warning("Fold %d: Model %s failed to fit: %s", fold_i, name, e)
                oof_preds[name][v_idx] = np.nan

    oof_df = pd.DataFrame({name: oof_preds[name][val_indices] for name in model_names}, index=val_indices)
    y_val_series = pd.Series(y_arr[val_indices], index=val_indices, name="y_true")
    
    return oof_df, y_val_series


# ===================================================================
# Weighted Blend (NNLS)
# ===================================================================

class WeightedBlendRegressor(BaseEstimator, RegressorMixin):
    """Ensemble that combines base models via non-negative normalized weights."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {}
        self.feature_names_: List[str] = []

    def fit(self, X_oof: pd.DataFrame, y_true: Union[pd.Series, np.ndarray]) -> "WeightedBlendRegressor":
        """Compute optimal non-negative least squares weights that sum to 1."""
        self.feature_names_ = list(X_oof.columns)
        X_mat = X_oof.values
        y_vec = np.asarray(y_true)

        # Drop any row with NaNs
        valid_rows = ~np.isnan(X_mat).any(axis=1) & ~np.isnan(y_vec)
        X_clean = X_mat[valid_rows]
        y_clean = y_vec[valid_rows]

        # NNLS: min ||X w - y||^2  s.t. w >= 0
        w, _ = nnls(X_clean, y_clean)
        w_sum = np.sum(w)
        if w_sum > 0:
            w_norm = w / w_sum
        else:
            w_norm = np.ones(len(w)) / len(w)

        self.weights = {col: float(w_norm[i]) for i, col in enumerate(self.feature_names_)}
        return self

    def predict(self, base_predictions: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Combine predictions using fitted weights."""
        if isinstance(base_predictions, pd.DataFrame):
            preds = np.zeros(len(base_predictions))
            for col, wt in self.weights.items():
                if col in base_predictions.columns:
                    preds += wt * base_predictions[col].values
            return preds
        elif isinstance(base_predictions, np.ndarray):
            w_vec = np.array([self.weights[c] for c in self.feature_names_])
            return base_predictions @ w_vec
        raise ValueError("Unsupported input format for predictions.")


# ===================================================================
# Stacking Ensemble (Ridge Meta-Learner)
# ===================================================================

class StackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking meta-learner using Ridge regression over base model predictions."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.meta_learner_ = Ridge(alpha=self.alpha, fit_intercept=True, positive=True)
        self.feature_names_: List[str] = []

    def fit(self, X_oof: pd.DataFrame, y_true: Union[pd.Series, np.ndarray]) -> "StackingRegressor":
        self.feature_names_ = list(X_oof.columns)
        X_mat = X_oof.values
        y_vec = np.asarray(y_true)

        valid_rows = ~np.isnan(X_mat).any(axis=1) & ~np.isnan(y_vec)
        self.meta_learner_.fit(X_mat[valid_rows], y_vec[valid_rows])
        return self

    def predict(self, base_predictions: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(base_predictions, pd.DataFrame):
            # Ensure columns order
            X_in = base_predictions[self.feature_names_].values
        else:
            X_in = base_predictions
        return self.meta_learner_.predict(X_in)


# ===================================================================
# Decision Rule: Ensemble vs Single Model
# ===================================================================

def evaluate_ensemble_acceptance(
    y_true: np.ndarray,
    ensemble_preds: np.ndarray,
    best_single_preds: np.ndarray,
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """Test whether ensemble significantly outperforms best single model.

    PRD Rule: Accept ensemble only if it beats the best single model by a non-trivial margin;
    otherwise choose best single model (simplicity wins ties).
    """
    comp = paired_bootstrap_test(
        y_true=y_true,
        y_pred_a=ensemble_preds,
        y_pred_b=best_single_preds,
        metric_fn=mean_absolute_error,
        n_boot=1000,
        seed=42,
    )
    # comp['diff'] = MAE(ensemble) - MAE(best_single). Negative means ensemble is better!
    mae_diff = comp["diff"]
    improvement = -mae_diff  # positive is better

    accept = bool(improvement > threshold and comp["ci_upper"] < 0)

    return {
        "ensemble_mae": comp["score_a"],
        "best_single_mae": comp["score_b"],
        "improvement": improvement,
        "ci_lower": comp["ci_lower"],
        "ci_upper": comp["ci_upper"],
        "p_value": comp["p_value"],
        "accepted": accept,
        "reason": (
            f"Ensemble beats single model by {improvement:.4f} MAE (95% CI: [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}])"
            if accept
            else f"Ensemble does not beat single model by >= {threshold} margin with statistically significant CI."
        ),
    }

