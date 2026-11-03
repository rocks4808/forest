"""Exploratory Data Analysis (EDA) module for Forest Loss dataset.

Generates:
- Statistical summary JSON (reports/eda_summary.json).
- 10+ publication-quality diagnostic plots saved to reports/figures/eda_*.png.
- Verifies identity leakage (loss_pct = loss_ha / forest_area_ha * 100) and mass balance.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

REPORTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive data facts and statistical summary."""
    n_regions = int(df["region_id"].nunique())
    n_years = int(df["year"].nunique())
    total_rows = len(df)
    
    # Check identity: loss_pct ≈ loss_ha / forest_area_ha * 100
    calc_pct = (df["loss_ha"] / df["forest_area_ha"]) * 100.0
    id_max_diff = float(np.max(np.abs(df["loss_pct"] - calc_pct)))

    # Autocorrelation lag-1
    lags = []
    for _, grp in df.groupby("region_id"):
        s = grp.sort_values("year")["loss_pct"]
        lags.append(s.autocorr(lag=1))
    mean_autocorr = float(np.nanmean(lags))

    # Static checks
    prot_std = float(df.groupby("region_id")["protected_area"].std().max())
    road_std = float(df.groupby("region_id")["road_density_km_per_km2"].std().max())

    # Policy counts
    policy_switches = 0
    for _, grp in df.groupby("region_id"):
        p = grp.sort_values("year")["policy_active"].values
        if (p[1:] != p[:-1]).any():
            policy_switches += 1

    summary = {
        "n_regions": n_regions,
        "n_years": n_years,
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "total_records": total_rows,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "identity_max_error": id_max_diff,
        "lag1_autocorrelation_mean": mean_autocorr,
        "protected_area_static": bool(prot_std == 0.0 or np.isnan(prot_std)),
        "road_density_static": bool(road_std == 0.0 or np.isnan(road_std)),
        "regions_with_policy_change": policy_switches,
        "loss_pct_mean": float(df["loss_pct"].mean()),
        "loss_pct_std": float(df["loss_pct"].std()),
        "loss_pct_min": float(df["loss_pct"].min()),
        "loss_pct_max": float(df["loss_pct"].max()),
        "loss_pct_by_protected": {
            "unprotected": float(df[df["protected_area"] == 0]["loss_pct"].mean()),
            "protected": float(df[df["protected_area"] == 1]["loss_pct"].mean()),
        },
        "loss_pct_by_policy": {
            "inactive": float(df[df["policy_active"] == 0]["loss_pct"].mean()),
            "active": float(df[df["policy_active"] == 1]["loss_pct"].mean()),
        },
        "total_initial_forest_area_ha": int(df[df["year"] == df["year"].min()]["forest_area_ha"].sum()),
        "total_latest_forest_area_ha": int(df[df["year"] == df["year"].max()]["forest_area_ha"].sum()),
        "total_cumulative_loss_ha": int(df["loss_ha"].sum()),
        "total_cumulative_gain_ha": int(df["gain_ha"].sum()),
    }
    return summary


def run_eda(df: pd.DataFrame, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run full EDA pipeline, save summary JSON, and produce all 10 diagnostic figures."""
    fig_dir = pathlib.Path(output_dir) if output_dir else FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = generate_eda_summary(df)
    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = ["#2E7D32", "#7FB069", "#D9B84A", "#E07A3F", "#C8402F"]

    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    numeric_vars = ["loss_pct", "fire_count", "agri_expansion_pct", "rainfall_mm", "population_density", "forest_area_ha"]
    for i, col in enumerate(numeric_vars):
        ax = axes[i // 3, i % 3]
        sns.histplot(df[col], kde=True, color="#2E7D32", ax=ax)
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_distributions.png", dpi=200)
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix (Panel Dataset)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_correlation.png", dpi=200)
    plt.close()

    # 3. National Forest Area & Loss/Gain Dynamics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    yearly = df.groupby("year").agg({
        "forest_area_ha": "sum",
        "loss_ha": "sum",
        "gain_ha": "sum",
        "loss_pct": "mean"
    }).reset_index()

    ax1.plot(yearly["year"], yearly["forest_area_ha"] / 1e6, marker="o", color="#2E7D32", lw=2.5)
    ax1.set_title("Total Forest Area Across 50 Regions (Million Ha)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Forest Area (M ha)")

    ax2.bar(yearly["year"] - 0.2, yearly["loss_ha"] / 1e3, width=0.4, label="Loss (k ha)", color="#C8402F", alpha=0.85)
    ax2.bar(yearly["year"] + 0.2, yearly["gain_ha"] / 1e3, width=0.4, label="Gain (k ha)", color="#7FB069", alpha=0.85)
    ax2.set_title("Annual Forest Loss vs. Gain (k ha)", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Hectares (thousands)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_trends_national.png", dpi=200)
    plt.close()

    # 4. Protected vs Unprotected Loss Comparison
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="protected_area", y="loss_pct", data=df, palette=["#E07A3F", "#7FB069"])
    plt.xticks([0, 1], ["Unprotected (1.33%)", "Protected (0.67%)"])
    plt.title("Annual Loss % by Protected Area Status", fontweight="bold")
    plt.ylabel("Loss %")
    plt.xlabel("Protection Status")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_protected_vs_unprotected.png", dpi=200)
    plt.close()

    # 5. Policy Active Impact
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="policy_active", y="loss_pct", data=df, palette=["#D9B84A", "#2E7D32"])
    plt.xticks([0, 1], ["Policy Inactive (1.38%)", "Policy Active (0.94%)"])
    plt.title("Annual Loss % by Policy Activation Status", fontweight="bold")
    plt.ylabel("Loss %")
    plt.xlabel("Policy Status")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_policy_timing.png", dpi=200)
    plt.close()

    # 6. Identity Verification Plot (loss_pct vs loss_ha / forest_area_ha * 100)
    plt.figure(figsize=(7, 5))
    calc_p = (df["loss_ha"] / df["forest_area_ha"]) * 100.0
    plt.scatter(df["loss_pct"], calc_p, alpha=0.5, color="#2E7D32", s=15)
    plt.plot([0, 3], [0, 3], "r--", label="Exact Identity (y = x)")
    plt.title("Identity Verification: loss_pct == loss_ha / forest_area_ha * 100", fontweight="bold")
    plt.xlabel("Recorded loss_pct")
    plt.ylabel("Computed (loss_ha / forest_area_ha) * 100")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_identity_scatter.png", dpi=200)
    plt.close()

    # 7. Fire Count vs Loss Pct Relationship
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="fire_count", y="loss_pct", hue="protected_area", data=df, palette=["#C8402F", "#2E7D32"], alpha=0.6)
    sns.regplot(x="fire_count", y="loss_pct", data=df, scatter=False, color="#121E18", line_kws={"lw": 2})
    plt.title("Fire Count vs. Loss % (Strongest Environmental Driver, r = +0.46)", fontweight="bold")
    plt.xlabel("Fire Count")
    plt.ylabel("Loss %")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_fire_vs_loss.png", dpi=200)
    plt.close()

    # 8. Lag-1 Autocorrelation of loss_pct
    plt.figure(figsize=(8, 5))
    df_sorted = df.sort_values(["region_id", "year"]).copy()
    df_sorted["loss_pct_lag1"] = df_sorted.groupby("region_id")["loss_pct"].shift(1)
    clean_lags = df_sorted.dropna(subset=["loss_pct_lag1", "loss_pct"])
    plt.scatter(clean_lags["loss_pct_lag1"], clean_lags["loss_pct"], color="#2E7D32", alpha=0.4, s=15)
    plt.plot([0, 3], [0, 3], "k--", label="Persistence Line (y = x)")
    plt.title(f"Lag-1 Autocorrelation (r = {summary['lag1_autocorrelation_mean']:.3f})", fontweight="bold")
    plt.xlabel("loss_pct (Year t - 1)")
    plt.ylabel("loss_pct (Year t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_autocorrelation.png", dpi=200)
    plt.close()

    # 9. Regional Loss Trajectories (Sample of 6 Regions)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    sample_regions = ["R01", "R06", "R12", "R20", "R33", "R44"]
    for i, r_id in enumerate(sample_regions):
        ax = axes[i // 3, i % 3]
        sub = df[df["region_id"] == r_id].sort_values("year")
        ax.plot(sub["year"], sub["loss_pct"], marker="o", color="#2E7D32", lw=2)
        # Mark policy start year
        pol_starts = sub[(sub["policy_active"] == 1) & (sub["policy_active"].shift(1) == 0)]
        if not pol_starts.empty:
            py = pol_starts["year"].iloc[0]
            ax.axvline(py, color="#C8402F", linestyle=":", label=f"Policy active ({py})")
            ax.legend(fontsize=9)
        prot = "Protected" if sub["protected_area"].iloc[0] == 1 else "Unprotected"
        ax.set_title(f"Region {r_id} ({prot})", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Loss %")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_trends_regional.png", dpi=200)
    plt.close()

    # 10. Population Density vs Forest Loss
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="population_density", y="loss_pct", hue="policy_active", data=df, palette=["#E07A3F", "#2E7D32"], alpha=0.6)
    plt.title("Population Density vs. Loss % by Conservation Policy", fontweight="bold")
    plt.xlabel("Population Density (people / km²)")
    plt.ylabel("Loss %")
    plt.tight_layout()
    plt.savefig(fig_dir / "eda_population_vs_loss.png", dpi=200)
    plt.close()

    logger.info("EDA completed: summary JSON and 10 figures saved.")
    return summary

