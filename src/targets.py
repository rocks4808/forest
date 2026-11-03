"""
targets.py – Build regression and classification targets for forest-loss prediction.

Panel data layout: 50 regions × 20 years (2005-2024).
Each row is region *r* at feature-year *t*.  Targets describe year *t + horizon*.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
THRESHOLD_FILE = MODELS_DIR / "thresholds.json"

SEED = 42


# ---------------------------------------------------------------------------
# Regression targets
# ---------------------------------------------------------------------------
def build_regression_targets(
    df: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Add next-year regression target columns to *df* (returns a copy).

    Columns added
    -------------
    - ``y_loss_pct_next``  : ``loss_pct`` at *t + horizon*  (primary regression target)
    - ``y_loss_ha_next``   : ``loss_ha``  at *t + horizon*
    - ``delta``            : ``y_loss_pct_next - loss_pct``  (residual / delta-learning target)
    - ``log_ratio``        : ``log(y_loss_pct_next / loss_pct)`` where ``loss_pct > 0``
    - ``target_year``      : ``year + horizon``  (the year described by the target)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``region_id``, ``year``, ``loss_pct``, and ``loss_ha``.
    horizon : int, default 1
        How many years ahead the target looks.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the target columns appended.
    """
    out = df.copy()
    out = out.sort_values(["region_id", "year"]).reset_index(drop=True)

    # For each region, shift the loss columns backwards by `horizon` so that
    # row *t* gets the value from row *t + horizon*.
    out["y_loss_pct_next"] = out.groupby("region_id")["loss_pct"].shift(-horizon)
    out["y_loss_ha_next"] = out.groupby("region_id")["loss_ha"].shift(-horizon)

    # Delta target: change in loss_pct
    out["delta"] = out["y_loss_pct_next"] - out["loss_pct"]

    # Log-ratio target (only defined where current loss_pct > 0)
    safe_mask = out["loss_pct"] > 0
    out["log_ratio"] = np.nan
    out.loc[safe_mask, "log_ratio"] = np.log(
        out.loc[safe_mask, "y_loss_pct_next"] / out.loc[safe_mask, "loss_pct"]
    )

    # Convenience column used by splits.py
    out["target_year"] = out["year"] + horizon

    return out


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
def compute_class_thresholds(
    y_train: pd.Series,
    mode: str = "ternary",
) -> Dict[str, float]:
    """Compute classification thresholds from *training data only*.

    Parameters
    ----------
    y_train : pd.Series
        ``y_loss_pct_next`` values for training rows (NaNs are dropped internally).
    mode : str, ``'ternary'`` | ``'binary'``
        * ``'ternary'`` → three classes (Low / Medium / High) split at the
          33rd and 67th percentiles of ``y_train``.
        * ``'binary'`` → two classes (0 / 1) with a fixed threshold of 1.5;
          if fewer than 25 % of training samples exceed 1.5 the top-quartile
          (75th percentile) is used instead.

    Returns
    -------
    dict
        Keys depend on *mode*:
        * ``'ternary'``: ``{"low_upper": float, "high_lower": float, "mode": "ternary"}``
        * ``'binary'``:  ``{"threshold": float, "mode": "binary"}``
    """
    y_clean: pd.Series = y_train.dropna()

    if mode == "ternary":
        low_upper: float = float(np.percentile(y_clean, 33))
        high_lower: float = float(np.percentile(y_clean, 67))
        thresholds = {
            "low_upper": low_upper,
            "high_lower": high_lower,
            "mode": "ternary",
        }
    elif mode == "binary":
        fixed_threshold = 1.5
        frac_above = float((y_clean >= fixed_threshold).mean())
        if frac_above >= 0.25:
            threshold = fixed_threshold
        else:
            threshold = float(np.percentile(y_clean, 75))
        thresholds = {
            "threshold": threshold,
            "mode": "binary",
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'ternary' or 'binary'.")

    # Persist thresholds to disk
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)

    return thresholds


# ---------------------------------------------------------------------------
# Classification target
# ---------------------------------------------------------------------------
def build_classification_targets(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    mode: str = "ternary",
) -> pd.DataFrame:
    """Add a ``severity_next`` column to *df* (returns a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain ``y_loss_pct_next``.
    thresholds : dict
        Output of :func:`compute_class_thresholds`.
    mode : str
        ``'ternary'`` or ``'binary'``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``severity_next`` appended.

    Encoding
    --------
    * *ternary*: 0 = Low, 1 = Medium, 2 = High
    * *binary*:  0 = Below threshold, 1 = At or above threshold
    """
    out = df.copy()

    if mode == "ternary":
        low_upper = thresholds["low_upper"]
        high_lower = thresholds["high_lower"]

        conditions = [
            out["y_loss_pct_next"] <= low_upper,
            out["y_loss_pct_next"] <= high_lower,
            out["y_loss_pct_next"] > high_lower,
        ]
        choices = [0, 1, 2]
        out["severity_next"] = np.select(
            conditions,
            choices,
            default=np.nan,
        )
    elif mode == "binary":
        threshold = thresholds["threshold"]
        out["severity_next"] = np.where(
            out["y_loss_pct_next"].notna(),
            (out["y_loss_pct_next"] >= threshold).astype(int),
            np.nan,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'ternary' or 'binary'.")

    # Rows with NaN target keep NaN severity
    out.loc[out["y_loss_pct_next"].isna(), "severity_next"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def build_all_targets(
    df: pd.DataFrame,
    train_mask: Optional[pd.Series] = None,
    mode: str = "ternary",
    horizon: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build every target column in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Raw panel data with at least ``region_id``, ``year``,
        ``loss_pct``, and ``loss_ha``.
    train_mask : pd.Series[bool] | None
        Boolean mask (aligned to *df* **after** regression targets are added)
        identifying training rows.  If ``None`` a default temporal mask is
        used: rows whose ``target_year <= 2021``.
    mode : str
        ``'ternary'`` or ``'binary'`` (forwarded to classification helpers).
    horizon : int
        Forecast horizon in years (default 1).

    Returns
    -------
    (pd.DataFrame, dict)
        DataFrame with all target columns and the threshold dictionary.
    """
    # 1. Regression targets
    out = build_regression_targets(df, horizon=horizon)

    # 2. Determine training rows for threshold computation
    if train_mask is None:
        # Default: target_year <= 2021  (features up to year 2020 for h=1)
        train_mask = out["target_year"] <= 2021

    # Align mask index with `out`
    if not train_mask.index.equals(out.index):
        train_mask = train_mask.reindex(out.index, fill_value=False)

    y_train = out.loc[train_mask, "y_loss_pct_next"]

    # 3. Classification thresholds (from training data only)
    thresholds = compute_class_thresholds(y_train, mode=mode)

    # 4. Classification target (applied to ALL rows, but thresholds come from train)
    out = build_classification_targets(out, thresholds, mode=mode)

    return out, thresholds
