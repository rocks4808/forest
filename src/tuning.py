"""Optuna hyperparameter tuning module for the Forest Loss Prediction pipeline.

Uses TPE sampler with seed=42 and expanding-window cross-validation.
Objective is mean CV MAE (regression) or mean CV Macro-F1 (classification).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import ElasticNet, Ridge, RidgeClassifier
from sklearn.metrics import f1_score, mean_absolute_error

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

from src.config import SEED

logger = logging.getLogger(__name__)

# Reduce Optuna noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

_REGRESSION_CONSTRUCTORS: Dict[str, Any] = {
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
    "RandomForest": RandomForestRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "GradientBoosting": GradientBoostingRegressor,
}

_CLASSIFICATION_CONSTRUCTORS: Dict[str, Any] = {
    "Ridge": RidgeClassifier,
    "RandomForest": RandomForestClassifier,
    "ExtraTrees": ExtraTreesClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

if xgb is not None:
    from xgboost import XGBRegressor, XGBClassifier
    _REGRESSION_CONSTRUCTORS["XGBoost"] = XGBRegressor
    _CLASSIFICATION_CONSTRUCTORS["XGBoost"] = XGBClassifier

if lgb is not None:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _REGRESSION_CONSTRUCTORS["LightGBM"] = LGBMRegressor
    _CLASSIFICATION_CONSTRUCTORS["LightGBM"] = LGBMClassifier


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------


def get_search_space(model_name: str, trial: optuna.Trial) -> dict:
    """Get hyperparameter search space for a model.

    Parameters
    ----------
    model_name : str
        Name of the model. One of: Ridge, ElasticNet, RandomForest,
        ExtraTrees, XGBoost, LightGBM, GradientBoosting.
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters.

    Returns
    -------
    dict
        Dictionary of hyperparameter name → suggested value.

    Raises
    ------
    ValueError
        If ``model_name`` is not recognized.
    """
    if model_name == "Ridge":
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
        }

    if model_name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "max_iter": 5000,
        }

    if model_name == "RandomForest":
        max_features_choice = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", "0.5", "0.8"]
        )
        max_features: Any = (
            float(max_features_choice) if max_features_choice in ("0.5", "0.8")
            else max_features_choice
        )
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": max_features,
        }

    if model_name == "ExtraTrees":
        max_features_choice = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", "0.5", "0.8"]
        )
        max_features = (
            float(max_features_choice) if max_features_choice in ("0.5", "0.8")
            else max_features_choice
        )
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": max_features,
        }

    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10.0, log=True),
        }

    if model_name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    raise ValueError(
        f"Unknown model_name={model_name!r}. Supported: Ridge, ElasticNet, "
        f"RandomForest, ExtraTrees, XGBoost, LightGBM, GradientBoosting."
    )


# ---------------------------------------------------------------------------
# Build model from params
# ---------------------------------------------------------------------------


def _build_model(
    model_name: str,
    params: dict,
    task: str = "regression",
    seed: int = SEED,
) -> Any:
    """Instantiate a scikit-learn compatible model with given params.

    Parameters
    ----------
    model_name : str
        Name of the model.
    params : dict
        Hyperparameters from ``get_search_space``.
    task : str
        ``'regression'`` or ``'classification'``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    model
        Instantiated scikit-learn compatible estimator.
    """
    constructors = (
        _REGRESSION_CONSTRUCTORS if task == "regression"
        else _CLASSIFICATION_CONSTRUCTORS
    )
    cls = constructors[model_name]

    full_params = dict(params)

    # Add seed / random_state
    if model_name in ("Ridge", "ElasticNet"):
        # Ridge / ElasticNet don't use random_state for fit
        pass
    elif model_name == "XGBoost":
        full_params["random_state"] = seed
        full_params["verbosity"] = 0
        full_params["n_jobs"] = 1
    elif model_name == "LightGBM":
        full_params["random_state"] = seed
        full_params["verbosity"] = -1
        full_params["n_jobs"] = 1
        full_params["force_col_wise"] = True
    elif model_name in ("RandomForest", "ExtraTrees"):
        full_params["random_state"] = seed
        full_params["n_jobs"] = 1
    else:
        full_params["random_state"] = seed

    return cls(**full_params)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _objective(
    trial: optuna.Trial,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    task: str,
    seed: int,
) -> float:
    """Optuna objective function.

    Trains a model with the suggested hyperparameters on each
    expanding-window CV split, evaluates on the validation fold,
    and returns the mean score across folds.

    For regression: returns mean MAE (lower is better → minimize).
    For classification: returns mean Macro-F1 (higher is better → maximize).

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    model_name : str
        Name of the model to tune.
    X : pd.DataFrame
        Full feature matrix (all folds combined).
    y : pd.Series
        Full target series (aligned with X).
    cv_splits : list
        List of (train_indices, val_indices) tuples from expanding-window CV.
    task : str
        ``'regression'`` or ``'classification'``.
    seed : int
        Random seed.

    Returns
    -------
    float
        Mean CV score (MAE for regression, Macro-F1 for classification).
    """
    params = get_search_space(model_name, trial)
    scores: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = _build_model(model_name, params, task=task, seed=seed)

        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.warning(
                "Trial %d fold %d failed for %s: %s",
                trial.number, fold_idx, model_name, exc,
            )
            raise optuna.TrialPruned(f"Fit failed: {exc}")

        preds = model.predict(X_val)

        if task == "regression":
            score = mean_absolute_error(y_val, preds)
        else:
            score = f1_score(y_val, preds, average="macro", zero_division=0)

        scores.append(score)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    n_trials: int = 80,
    task: str = "regression",
    seed: int = SEED,
) -> Tuple[dict, float]:
    """Tune a single model with Optuna TPE sampler.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. ``'XGBoost'``, ``'LightGBM'``, ``'Ridge'``).
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series (aligned with X).
    cv_splits : list
        List of ``(train_idx, val_idx)`` from expanding-window CV.
    n_trials : int
        Number of Optuna trials to run. Default 80.
    task : str
        ``'regression'`` (minimize MAE) or ``'classification'`` (maximize F1).
    seed : int
        Random seed for TPE sampler.

    Returns
    -------
    tuple[dict, float]
        ``(best_params, best_score)`` where best_score is the best mean CV
        MAE (regression) or Macro-F1 (classification).
    """
    direction = "minimize" if task == "regression" else "maximize"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=f"{model_name}_{task}",
    )

    study.optimize(
        lambda trial: _objective(
            trial, model_name, X, y, cv_splits, task, seed
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_score = study.best_value

    # Post-process categorical max_features back from string
    if "max_features" in best_params:
        mf = best_params["max_features"]
        if mf in ("0.5", "0.8"):
            best_params["max_features"] = float(mf)

    logger.info(
        "Tuned %s (%s): best %s = %.6f | best params = %s",
        model_name,
        task,
        "MAE" if task == "regression" else "Macro-F1",
        best_score,
        best_params,
    )

    return best_params, best_score


def tune_all_models(
    model_names: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    n_trials: int = 80,
    task: str = "regression",
    seed: int = SEED,
) -> dict:
    """Tune all specified models and return a results dictionary.

    Parameters
    ----------
    model_names : list[str]
        List of model name strings to tune.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series.
    cv_splits : list
        List of ``(train_idx, val_idx)`` from expanding-window CV.
    n_trials : int
        Number of Optuna trials per model. Default 80.
    task : str
        ``'regression'`` or ``'classification'``.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{model_name: {'params': dict, 'score': float}}`` for each model.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name in model_names:
        logger.info("Tuning %s (%d trials)...", name, n_trials)
        try:
            best_params, best_score = tune_model(
                model_name=name,
                X=X,
                y=y,
                cv_splits=cv_splits,
                n_trials=n_trials,
                task=task,
                seed=seed,
            )
            results[name] = {"params": best_params, "score": best_score}
        except Exception as exc:
            logger.error("Failed to tune %s: %s", name, exc)
            results[name] = {"params": {}, "score": float("inf") if task == "regression" else 0.0}

    # Sort by best score
    if task == "regression":
        ranked = sorted(results.items(), key=lambda kv: kv[1]["score"])
    else:
        ranked = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)

    logger.info("=== Tuning Results (%s) ===", task)
    for rank, (name, info) in enumerate(ranked, start=1):
        metric_name = "MAE" if task == "regression" else "Macro-F1"
        logger.info(
            "  #%d  %-20s  %s = %.6f", rank, name, metric_name, info["score"]
        )

    return results


def build_tuned_model(
    model_name: str,
    params: dict,
    task: str = "regression",
    seed: int = SEED,
) -> Any:
    """Build a model instance with tuned hyperparameters.

    Convenience wrapper around ``_build_model`` exposed for external use.

    Parameters
    ----------
    model_name : str
        Model identifier.
    params : dict
        Hyperparameters (e.g. from ``tune_model``).
    task : str
        ``'regression'`` or ``'classification'``.
    seed : int
        Random seed.

    Returns
    -------
    model
        Instantiated (unfitted) scikit-learn compatible estimator.
    """
    return _build_model(model_name, params, task=task, seed=seed)
