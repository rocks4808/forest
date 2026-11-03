"""Data loader and validator for the Forest Loss ML pipeline.

This module loads the forest_loss_trend.csv dataset, performs comprehensive
schema, range, uniqueness, identity, and mass-balance validations, and
returns a clean, sorted DataFrame ready for downstream feature engineering.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: list[str] = [
    "region_id",
    "year",
    "forest_area_ha",
    "loss_ha",
    "gain_ha",
    "loss_pct",
    "population_density",
    "road_density_km_per_km2",
    "agri_expansion_pct",
    "rainfall_mm",
    "fire_count",
    "protected_area",
    "policy_active",
]

EXPECTED_DTYPES: Dict[str, type] = {
    "region_id": str,
    "year": np.int64,
    "forest_area_ha": np.int64,
    "loss_ha": np.int64,
    "gain_ha": np.int64,
    "loss_pct": np.float64,
    "population_density": np.float64,
    "road_density_km_per_km2": np.float64,
    "agri_expansion_pct": np.float64,
    "rainfall_mm": np.int64,
    "fire_count": np.int64,
    "protected_area": np.int64,
    "policy_active": np.int64,
}

YEAR_MIN: int = 2005
YEAR_MAX: int = 2024
EXPECTED_YEARS: list[int] = list(range(YEAR_MIN, YEAR_MAX + 1))
NUM_REGIONS: int = 50
NUM_YEARS: int = 20
EXPECTED_ROWS: int = NUM_REGIONS * NUM_YEARS  # 1000

LOSS_PCT_TOL: float = 0.01   # absolute tolerance for loss_pct identity check
MASS_BALANCE_TOL: float = 2.0  # absolute tolerance in hectares


# ---------------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------------


def _check_columns(df: pd.DataFrame) -> bool:
    """Check that all 13 expected columns exist in the DataFrame.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if all expected columns are present, False otherwise.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing:
        logger.error("Missing columns: %s", missing)
    if extra:
        logger.warning("Extra columns (ignored): %s", extra)
    return len(missing) == 0


def _check_dtypes(df: pd.DataFrame) -> bool:
    """Verify that each column has the expected dtype family.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if all dtypes match expectations, False otherwise.
    """
    ok = True
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual_kind = df[col].dtype.kind
        if expected is str:
            # object or string dtype
            if actual_kind not in ("O", "U"):
                logger.error(
                    "Column '%s' expected str-like dtype, got %s", col, df[col].dtype
                )
                ok = False
        elif expected in (np.int64, int):
            if actual_kind not in ("i", "u"):
                logger.error(
                    "Column '%s' expected int dtype, got %s", col, df[col].dtype
                )
                ok = False
        elif expected in (np.float64, float):
            if actual_kind != "f":
                logger.error(
                    "Column '%s' expected float dtype, got %s", col, df[col].dtype
                )
                ok = False
    return ok


def _check_no_missing(df: pd.DataFrame) -> bool:
    """Check that no values are missing (NaN / null).

    Args:
        df: The loaded DataFrame.

    Returns:
        True if there are no missing values, False otherwise.
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        logger.error(
            "Found %d missing values:\n%s",
            total_missing,
            missing_counts[missing_counts > 0].to_string(),
        )
        return False
    return True


def _check_no_duplicates(df: pd.DataFrame) -> bool:
    """Check that there are no fully duplicate rows.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if there are no duplicate rows, False otherwise.
    """
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        logger.error("Found %d fully duplicate rows.", n_dups)
        return False
    return True


def _check_unique_region_year(df: pd.DataFrame) -> bool:
    """Check that (region_id, year) pairs are unique.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if all (region_id, year) pairs are unique, False otherwise.
    """
    n_dups = df.duplicated(subset=["region_id", "year"]).sum()
    if n_dups > 0:
        logger.error("Found %d duplicate (region_id, year) pairs.", n_dups)
        return False
    return True


def _check_year_range(df: pd.DataFrame) -> bool:
    """Check that year values fall within [2005, 2024].

    Args:
        df: The loaded DataFrame.

    Returns:
        True if all years are within the expected range, False otherwise.
    """
    oob = df[(df["year"] < YEAR_MIN) | (df["year"] > YEAR_MAX)]
    if len(oob) > 0:
        logger.error(
            "%d rows have year outside [%d, %d].", len(oob), YEAR_MIN, YEAR_MAX
        )
        return False
    return True


def _check_loss_pct_range(df: pd.DataFrame) -> bool:
    """Check that loss_pct values fall within [0, 100].

    Args:
        df: The loaded DataFrame.

    Returns:
        True if all loss_pct values are within [0, 100], False otherwise.
    """
    oob = df[(df["loss_pct"] < 0) | (df["loss_pct"] > 100)]
    if len(oob) > 0:
        logger.error("%d rows have loss_pct outside [0, 100].", len(oob))
        return False
    return True


def _check_binary_columns(df: pd.DataFrame) -> bool:
    """Check that protected_area and policy_active contain only 0 or 1.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if both columns are binary, False otherwise.
    """
    ok = True
    for col in ("protected_area", "policy_active"):
        unique_vals = set(df[col].unique())
        if not unique_vals.issubset({0, 1}):
            logger.error("Column '%s' has non-binary values: %s", col, unique_vals)
            ok = False
    return ok


def _check_consecutive_years(df: pd.DataFrame) -> bool:
    """Check that every region has exactly the years 2005-2024 (consecutive).

    Args:
        df: The loaded DataFrame.

    Returns:
        True if each region has exactly years 2005-2024, False otherwise.
    """
    ok = True
    expected_set = set(EXPECTED_YEARS)
    for region_id, grp in df.groupby("region_id"):
        actual_years = set(grp["year"].values)
        if actual_years != expected_set:
            missing_years = expected_set - actual_years
            extra_years = actual_years - expected_set
            logger.error(
                "Region '%s': missing years=%s, extra years=%s",
                region_id,
                missing_years or "none",
                extra_years or "none",
            )
            ok = False
    return ok


def _check_loss_pct_identity(df: pd.DataFrame) -> bool:
    """Check the identity: loss_pct ≈ loss_ha / forest_area_ha * 100.

    Uses an absolute tolerance of 0.01.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if the identity holds within tolerance for all rows, False otherwise.
    """
    computed_pct = df["loss_ha"] / df["forest_area_ha"] * 100.0
    diff = (df["loss_pct"] - computed_pct).abs()
    violations = diff > LOSS_PCT_TOL
    n_violations = violations.sum()
    if n_violations > 0:
        max_diff = diff.max()
        logger.error(
            "%d rows violate loss_pct identity (max diff=%.4f).",
            n_violations,
            max_diff,
        )
        return False
    return True


def _check_mass_balance(df: pd.DataFrame) -> bool:
    """Check mass balance: forest_area(t+1) ≈ forest_area(t) - loss(t) + gain(t).

    Uses an absolute tolerance of 2 ha.  Only consecutive year pairs within
    each region are checked.

    Args:
        df: The loaded DataFrame (must be sorted by region_id, year).

    Returns:
        True if mass balance holds within tolerance, False otherwise.
    """
    ok = True
    sorted_df = df.sort_values(["region_id", "year"]).reset_index(drop=True)
    n_violations = 0
    max_diff = 0.0

    for region_id, grp in sorted_df.groupby("region_id"):
        grp = grp.sort_values("year").reset_index(drop=True)
        for i in range(len(grp) - 1):
            area_t = grp.loc[i, "forest_area_ha"]
            loss_t = grp.loc[i, "loss_ha"]
            gain_t = grp.loc[i, "gain_ha"]
            area_next = grp.loc[i + 1, "forest_area_ha"]
            expected_next = area_t - loss_t + gain_t
            diff = abs(area_next - expected_next)
            if diff > MASS_BALANCE_TOL:
                n_violations += 1
                max_diff = max(max_diff, diff)

    if n_violations > 0:
        logger.error(
            "%d region-year transitions violate mass balance (max diff=%.2f ha).",
            n_violations,
            max_diff,
        )
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_all_checks(df: pd.DataFrame) -> Dict[str, bool]:
    """Run all validation checks on the DataFrame.

    Each check is independent; all checks are always executed so the caller
    gets a complete picture of data quality.

    Args:
        df: The loaded DataFrame.

    Returns:
        Dictionary mapping check name → True (passed) / False (failed).
    """
    results: Dict[str, bool] = {
        "columns_present": _check_columns(df),
        "dtypes_correct": _check_dtypes(df),
        "no_missing_values": _check_no_missing(df),
        "no_duplicate_rows": _check_no_duplicates(df),
        "unique_region_year": _check_unique_region_year(df),
        "year_in_range": _check_year_range(df),
        "loss_pct_in_range": _check_loss_pct_range(df),
        "binary_columns_valid": _check_binary_columns(df),
        "consecutive_years": _check_consecutive_years(df),
        "loss_pct_identity": _check_loss_pct_identity(df),
        "mass_balance": _check_mass_balance(df),
    }
    passed = sum(results.values())
    total = len(results)
    logger.info("Validation: %d / %d checks passed.", passed, total)
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        logger.info("  %s %s", symbol, name)
    return results


def load_and_validate(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load and validate the forest loss dataset.

    Reads ``forest_loss_trend.csv``, enforces schema / range / identity /
    mass-balance constraints, and returns a clean DataFrame sorted by
    ``region_id`` and ``year``.

    Args:
        data_path: Absolute or relative path to the CSV file.  Defaults to
            ``<ROOT_DIR>/data/forest_loss_trend.csv``.

    Returns:
        A validated ``pd.DataFrame`` sorted by (region_id, year).

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValueError: If any critical validation check fails.
    """
    if data_path is None:
        data_path = os.path.join(ROOT_DIR, "data", "forest_loss_trend.csv")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, dtype={"region_id": str})

    # Sort deterministically
    df = df.sort_values(["region_id", "year"]).reset_index(drop=True)

    logger.info("Loaded %d rows × %d columns.", df.shape[0], df.shape[1])

    # Run all validation checks
    results = run_all_checks(df)

    # Raise on any failure
    failed_checks = [name for name, passed in results.items() if not passed]
    if failed_checks:
        raise ValueError(
            f"Data validation failed for {len(failed_checks)} check(s): "
            + ", ".join(failed_checks)
        )

    logger.info("All validation checks passed. Returning validated DataFrame.")
    return df


# ---------------------------------------------------------------------------
# CLI entry-point (for quick manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    validated_df = load_and_validate()
    print(f"\nDataFrame shape: {validated_df.shape}")
    print(validated_df.head(10))
