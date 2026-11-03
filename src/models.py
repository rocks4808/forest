"""
Model factory and wrappers for the forest loss prediction pipeline.

Provides a registry-based pattern for creating models by name,
with graceful fallback for optional dependencies (e.g., CatBoost).
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Optional CatBoost import
try:
    from catboost import CatBoostRegressor, CatBoostClassifier

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False
    logging.getLogger(__name__).warning(
        "CatBoost is not installed. CatBoost models will be unavailable."
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Baseline Models
# ---------------------------------------------------------------------------


class PersistenceBaseline:
    """Predicts y_next = current loss_pct (direct) or delta = 0 (delta target).

    For a direct target the prediction is simply ``X[persistence_col]``.
    For a delta target the persistence forecast is zero (no change), so the
    final prediction after adding back the current value is the same.

    Parameters
    ----------
    persistence_col : str
        Column in *X* that holds the most-recent known loss_pct value.
        Defaults to ``'loss_pct_lag1'``.
    """

    def __init__(self, persistence_col: str = "loss_pct_lag1") -> None:
        self.persistence_col: str = persistence_col
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PersistenceBaseline":
        """Store metadata; no real training needed.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must contain ``persistence_col``).
        y : np.ndarray
            Target values (unused, kept for API compatibility).

        Returns
        -------
        self
        """
        if self.persistence_col not in X.columns:
            raise ValueError(
                f"Persistence column '{self.persistence_col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the persistence forecast.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing ``persistence_col``.

        Returns
        -------
        np.ndarray
            Predictions equal to ``X[persistence_col]``.
        """
        if not self._is_fitted:
            raise RuntimeError("PersistenceBaseline has not been fitted yet.")
        if self.persistence_col not in X.columns:
            raise ValueError(
                f"Persistence column '{self.persistence_col}' not found in X."
            )
        return X[self.persistence_col].values.astype(np.float64)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Sklearn-compatible ``get_params``."""
        return {"persistence_col": self.persistence_col}

    def set_params(self, **params: Any) -> "PersistenceBaseline":
        """Sklearn-compatible ``set_params``."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"PersistenceBaseline(persistence_col='{self.persistence_col}')"


class RegionMeanBaseline:
    """Predicts the mean of ``loss_pct`` per region from training data.

    During ``fit`` the model memorises the per-region mean.  At ``predict``
    time it looks up the region for each row and returns the stored mean.
    Unseen regions fall back to the global training mean.

    Parameters
    ----------
    region_col : str
        Column name identifying the region.  Defaults to ``'subnational1'``.
    """

    def __init__(self, region_col: str = "region_id") -> None:
        self.region_col: str = region_col
        self._region_means: Dict[Any, float] = {}
        self._global_mean: float = 0.0
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RegionMeanBaseline":
        """Compute per-region means from the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must contain ``region_col``).
        y : np.ndarray
            Target values.

        Returns
        -------
        self
        """
        if self.region_col not in X.columns:
            raise ValueError(
                f"Region column '{self.region_col}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        y_arr = np.asarray(y, dtype=np.float64)
        self._global_mean = float(np.nanmean(y_arr))
        regions = X[self.region_col].values
        self._region_means = {}
        for region in np.unique(regions):
            mask = regions == region
            self._region_means[region] = float(np.nanmean(y_arr[mask]))
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-region mean predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing ``region_col``.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("RegionMeanBaseline has not been fitted yet.")
        if self.region_col not in X.columns:
            raise ValueError(
                f"Region column '{self.region_col}' not found in X."
            )
        regions = X[self.region_col].values
        preds = np.array(
            [self._region_means.get(r, self._global_mean) for r in regions],
            dtype=np.float64,
        )
        return preds

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"region_col": self.region_col}

    def set_params(self, **params: Any) -> "RegionMeanBaseline":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"RegionMeanBaseline(region_col='{self.region_col}')"


class LinearTrendBaseline:
    """Fits a per-region linear trend (loss_pct ~ year) and extrapolates.

    Parameters
    ----------
    region_col : str
        Column name identifying the region.
    year_col : str
        Column name for the year feature.
    """

    def __init__(
        self, region_col: str = "region_id", year_col: str = "year"
    ) -> None:
        self.region_col: str = region_col
        self.year_col: str = year_col
        # {region: (slope, intercept)}
        self._trends: Dict[Any, tuple] = {}
        self._global_slope: float = 0.0
        self._global_intercept: float = 0.0
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearTrendBaseline":
        """Fit per-region OLS trend lines.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``region_col`` and ``year_col``.
        y : np.ndarray
            Target values.

        Returns
        -------
        self
        """
        for col in (self.region_col, self.year_col):
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X.")

        y_arr = np.asarray(y, dtype=np.float64)
        years = X[self.year_col].values.astype(np.float64)
        regions = X[self.region_col].values

        # Global trend as fallback
        if len(years) > 1:
            coeffs = np.polyfit(years, y_arr, 1)
            self._global_slope, self._global_intercept = float(coeffs[0]), float(coeffs[1])
        else:
            self._global_slope = 0.0
            self._global_intercept = float(y_arr[0]) if len(y_arr) > 0 else 0.0

        self._trends = {}
        for region in np.unique(regions):
            mask = regions == region
            reg_years = years[mask]
            reg_y = y_arr[mask]
            if len(reg_years) >= 2 and np.std(reg_years) > 0:
                coeffs = np.polyfit(reg_years, reg_y, 1)
                self._trends[region] = (float(coeffs[0]), float(coeffs[1]))
            elif len(reg_years) >= 1:
                self._trends[region] = (0.0, float(np.mean(reg_y)))
            # else: will use global fallback

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Extrapolate the per-region linear trend.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``region_col`` and ``year_col``.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("LinearTrendBaseline has not been fitted yet.")
        for col in (self.region_col, self.year_col):
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in X.")

        regions = X[self.region_col].values
        years = X[self.year_col].values.astype(np.float64)
        preds = np.empty(len(X), dtype=np.float64)

        for i, (region, yr) in enumerate(zip(regions, years)):
            slope, intercept = self._trends.get(
                region, (self._global_slope, self._global_intercept)
            )
            preds[i] = slope * yr + intercept

        return preds

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"region_col": self.region_col, "year_col": self.year_col}

    def set_params(self, **params: Any) -> "LinearTrendBaseline":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"LinearTrendBaseline(region_col='{self.region_col}', "
            f"year_col='{self.year_col}')"
        )


# ---------------------------------------------------------------------------
# Default hyper-parameters for every model
# ---------------------------------------------------------------------------

_REGRESSION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "persistence": {},
    "region_mean": {},
    "linear_trend": {},
    "ridge": {"alpha": 1.0},
    "elasticnet": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000},
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 10,
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbose": -1,
    },
    "catboost": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "verbose": 0,
    },
    "mlp": {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "learning_rate_init": 0.001,
    },
}

_CLASSIFICATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "max_iter": 2000,
        "solver": "lbfgs",
        "C": 1.0,
    },
    "random_forest_clf": {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "n_jobs": -1,
    },
    "xgboost_clf": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "verbosity": 0,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    },
    "lightgbm_clf": {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "verbose": -1,
    },
    "catboost_clf": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "verbose": 0,
    },
}


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_regression_model(
    name: str, params: Dict[str, Any], seed: int
) -> Any:
    """Instantiate a regression model by canonical name.

    Parameters
    ----------
    name : str
        Canonical model name (lower-case, underscore-separated).
    params : dict
        Hyper-parameters to override defaults.
    seed : int
        Random seed.

    Returns
    -------
    model
        A fitted-API-compatible estimator.
    """
    defaults = _REGRESSION_DEFAULTS.get(name, {}).copy()
    defaults.update(params)

    if name == "persistence":
        return PersistenceBaseline(**defaults)
    if name == "region_mean":
        return RegionMeanBaseline(**defaults)
    if name == "linear_trend":
        return LinearTrendBaseline(**defaults)
    if name == "ridge":
        return Ridge(random_state=seed, **defaults)
    if name == "elasticnet":
        return ElasticNet(random_state=seed, **defaults)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, **defaults)
    if name == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, **defaults)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed, **defaults)
    if name == "xgboost":
        return XGBRegressor(random_state=seed, **defaults)
    if name == "lightgbm":
        return LGBMRegressor(random_state=seed, **defaults)
    if name == "catboost":
        if not _HAS_CATBOOST:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            )
        return CatBoostRegressor(random_seed=seed, **defaults)
    if name == "mlp":
        return MLPRegressor(random_state=seed, **defaults)

    raise ValueError(
        f"Unknown regression model '{name}'. "
        f"Available: {get_available_models('regression')}"
    )


def _build_classification_model(
    name: str, params: Dict[str, Any], seed: int
) -> Any:
    """Instantiate a classification model by canonical name.

    Parameters
    ----------
    name : str
        Canonical model name.
    params : dict
        Hyper-parameters to override defaults.
    seed : int
        Random seed.

    Returns
    -------
    model
        A classification estimator.
    """
    defaults = _CLASSIFICATION_DEFAULTS.get(name, {}).copy()
    defaults.update(params)

    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **defaults)
    if name == "random_forest_clf":
        return RandomForestClassifier(random_state=seed, **defaults)
    if name == "xgboost_clf":
        return XGBClassifier(random_state=seed, **defaults)
    if name == "lightgbm_clf":
        return LGBMClassifier(random_state=seed, **defaults)
    if name == "catboost_clf":
        if not _HAS_CATBOOST:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            )
        return CatBoostClassifier(random_seed=seed, **defaults)

    raise ValueError(
        f"Unknown classification model '{name}'. "
        f"Available: {get_available_models('classification')}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model(
    name: str,
    task: str = "regression",
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Any:
    """Get a model instance by name.

    Parameters
    ----------
    name : str
        Model identifier (e.g. ``'xgboost'``, ``'random_forest_clf'``).
    task : str
        ``'regression'`` or ``'classification'``.
    params : dict, optional
        Hyper-parameter overrides merged on top of defaults.
    seed : int
        Random seed used by the estimator.

    Returns
    -------
    model
        An estimator with scikit-learn compatible ``fit`` / ``predict`` API.

    Raises
    ------
    ValueError
        If *name* is unknown for the requested *task*.
    ImportError
        If *name* requires an optional dependency that is not installed.
    """
    params = params or {}
    name = name.strip().lower()

    if task == "regression":
        return _build_regression_model(name, params, seed)
    if task == "classification":
        return _build_classification_model(name, params, seed)

    raise ValueError(f"Unknown task '{task}'. Use 'regression' or 'classification'.")


def get_available_models(task: str = "regression") -> List[str]:
    """Return the list of available model names for a given task.

    Parameters
    ----------
    task : str
        ``'regression'`` or ``'classification'``.

    Returns
    -------
    list of str
        Canonical model names that can be passed to :func:`get_model`.
    """
    if task == "regression":
        names = [
            "persistence",
            "region_mean",
            "linear_trend",
            "ridge",
            "elasticnet",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "mlp",
        ]
        if _HAS_CATBOOST:
            names.append("catboost")
        return names

    if task == "classification":
        names = [
            "logistic_regression",
            "random_forest_clf",
            "xgboost_clf",
            "lightgbm_clf",
        ]
        if _HAS_CATBOOST:
            names.append("catboost_clf")
        return names

    raise ValueError(f"Unknown task '{task}'. Use 'regression' or 'classification'.")


def seed_average_predict(
    models: List[Any], X: pd.DataFrame
) -> np.ndarray:
    """Average predictions from multiple seed-variant (or ensemble) models.

    Parameters
    ----------
    models : list
        Fitted estimators that expose a ``predict(X)`` method.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Element-wise mean of all model predictions, shape ``(len(X),)``.

    Raises
    ------
    ValueError
        If the model list is empty.
    """
    if not models:
        raise ValueError("Model list is empty; cannot compute seed average.")

    preds = np.column_stack(
        [model.predict(X).astype(np.float64) for model in models]
    )
    return np.mean(preds, axis=1)
