"""Reusable UI components and editorial styling for Streamlit dashboard.

Follows the Deep Dark Forest visual identity:
- Palette: #0B1410 (bg), #121E18 (surface), #16241D (card), #24382D (border)
- Text: #E9E6DA (primary bone), #9FB0A4 (secondary), #6B7F73 (muted)
- Accent: #7FB069 (moss)
- Risk ramp: #7FB069 (Low) -> #D9B84A (Moderate) -> #E07A3F (High) -> #C8402F (Critical)
- Typography: Fraunces (serif headlines), IBM Plex Sans (body), IBM Plex Mono (numerals)
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

RISK_COLORS = {
    "Critical": "#C8402F",
    "High": "#E07A3F",
    "Moderate": "#D9B84A",
    "Low": "#7FB069",
}


def apply_custom_css() -> None:
    """Inject custom editorial CSS styling."""
    css_path = pathlib.Path(__file__).resolve().parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def top_nav_bar() -> None:
    """Render top product bar with status pill."""
    st.markdown(
        """
        <div class="top-nav-bar">
            <div class="brand-title">
                🌲 CanopyIntel &middot; Forest Loss & Degradation Hotspot Console
            </div>
            <div class="status-pill">
                Model v1.0 &bull; Seed 42 &bull; 2005&ndash;2024 Historical &bull; Forecast to 2030
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_headline(headline: str) -> None:
    """Render a computed editorial insight in the Fraunces serif typeface."""
    st.markdown(
        f"""
        <div class="insight-headline">
            {headline}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_caption(text: str) -> None:
    """Render a subtle 'what to notice' annotation caption under charts."""
    st.markdown(
        f"""
        <div class="chart-caption">
            &boxur; <strong>What to notice:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "neutral",  # 'good', 'warn', 'bad', 'neutral'
    sparkline_values: Optional[List[float]] = None,
    tooltip: Optional[str] = None,
) -> None:
    """Render an editorial KPI card with tabular numerals and optional inline sparkline."""
    delta_class = f"delta-{delta_type}" if delta else ""
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""

    spark_svg = ""
    if sparkline_values and len(sparkline_values) >= 2:
        vals = np.array(sparkline_values, dtype=float)
        v_min, v_max = vals.min(), vals.max()
        rng = v_max - v_min if v_max > v_min else 1.0
        w, h = 80, 24
        pts = []
        for idx, v in enumerate(vals):
            x = int(idx * (w / (len(vals) - 1)))
            y = int(h - ((v - v_min) / rng) * (h - 4) - 2)
            pts.append(f"{x},{y}")
        poly_pts = " ".join(pts)
        spark_color = "#7FB069" if delta_type in ["good", "neutral"] else ("#E07A3F" if delta_type == "warn" else "#C8402F")
        spark_svg = f"""
        <svg width="{w}" height="{h}" style="margin-top:6px; overflow:visible;">
            <polyline fill="none" stroke="{spark_color}" stroke-width="1.8" points="{poly_pts}" />
        </svg>
        """

    st.markdown(
        f"""
        <div class="kpi-container" {tooltip_attr}>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                {delta_html}
                {spark_svg}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tier_badge(tier: str) -> str:
    """Generate HTML badge string for risk tier."""
    cls = f"badge-{tier.lower()}"
    return f'<span class="{cls}">{tier}</span>'


def get_plotly_theme() -> go.layout.Template:
    """Shared editorial Plotly dark forest theme template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 30, 24, 0.4)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#E9E6DA", size=12),
        title=dict(font=dict(family="Fraunces, serif", size=16, color="#E9E6DA"), x=0.0, xanchor="left"),
        xaxis=dict(
            gridcolor="#24382D",
            zerolinecolor="#24382D",
            showline=True,
            linecolor="#24382D",
            tickfont=dict(family="IBM Plex Mono", size=11, color="#9FB0A4"),
            title=dict(font=dict(size=12, color="#9FB0A4")),
        ),
        yaxis=dict(
            gridcolor="#24382D",
            zerolinecolor="#24382D",
            showline=True,
            linecolor="#24382D",
            tickfont=dict(family="IBM Plex Mono", size=11, color="#9FB0A4"),
            title=dict(font=dict(size=12, color="#9FB0A4")),
        ),
        colorway=["#7FB069", "#D9B84A", "#E07A3F", "#C8402F", "#5DADE2", "#A569BD"],
        hoverlabel=dict(
            bgcolor="#16241D",
            bordercolor="#24382D",
            font=dict(family="IBM Plex Mono", color="#E9E6DA", size=11),
        ),
        legend=dict(
            bgcolor="rgba(22, 36, 29, 0.8)",
            bordercolor="#24382D",
            borderwidth=1,
            font=dict(size=11, color="#9FB0A4"),
        ),
    )
    return template


def render_schematic_cartogram(
    hotspots_df: pd.DataFrame,
    selected_region: Optional[str] = None,
    show_header: bool = False,
) -> None:
    """Render an interactive 10 x 5 schematic tile cartogram using Plotly Heatmap.
    
    Labeled honestly as schematic layout for visual comparison — not geographic coordinates.
    """
    if show_header:
        st.markdown("##### Schematic Regional Risk Cartogram (10 &times; 5 Grid)")
        st.caption("*Schematic layout for comparative screening — not geographic coordinates.*")

    sorted_df = hotspots_df.sort_values("region_id").reset_index(drop=True)

    cols = 10
    rows = 5

    z_matrix = np.zeros((rows, cols))
    text_matrix = []
    hover_matrix = []

    for r in range(rows):
        row_text = []
        row_hover = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(sorted_df):
                row = sorted_df.iloc[idx]
                r_id = str(row["region_id"])
                score = float(row["risk_score"])
                tier = str(row["tier"])
                pred_loss = float(row.get("predicted_loss_pct_next", 0.0))
                slope = float(row.get("slope_5yr", 0.0))

                y_idx = rows - 1 - r
                z_matrix[y_idx, c] = score

                sel_mark = " *" if r_id == selected_region else ""
                row_text.append(f"<b>{r_id}{sel_mark}</b><br>{score:.0f}")
                row_hover.append(
                    f"<b>Jurisdiction: {r_id}</b><br>"
                    f"Risk Score: <b>{score:.1f}</b> / 100<br>"
                    f"Severity Tier: <b>{tier}</b><br>"
                    f"Predicted Loss: <b>{pred_loss:.2f}%</b><br>"
                    f"5-Yr Trend: <b>{slope:+.3f}%/yr</b><extra></extra>"
                )
            else:
                row_text.append("")
                row_hover.append("")
        text_matrix.append(row_text)
        hover_matrix.append(row_hover)

    # Invert for bottom-to-top layout in Plotly
    text_matrix = text_matrix[::-1]
    hover_matrix = hover_matrix[::-1]

    # Risk ramp colorscale: Low (green) -> Mod (yellow) -> High (orange) -> Crit (red)
    risk_colorscale = [
        [0.0, "#1A2C23"],
        [0.25, "#3D5A47"],
        [0.45, "#7FB069"],
        [0.65, "#D9B84A"],
        [0.80, "#E07A3F"],
        [1.0, "#C8402F"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono", size=10, color="#E9E6DA"),
        hoverinfo="text",
        hovertext=hover_matrix,
        colorscale=risk_colorscale,
        zmin=0,
        zmax=100,
        xgap=4,
        ygap=4,
        showscale=False,
    ))

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_layout(
        template=get_plotly_theme(),
        height=280,
        margin=dict(l=4, r=4, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, width="stretch")


# ===================================================================
# Data and Artifact Loaders (Cached)
# ===================================================================

@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Load raw or processed historical dataset."""
    path = DATA_DIR / "forest_loss_trend.csv"
    if not path.exists():
        path = ROOT_DIR / "forest_loss_trend.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_hotspots_data() -> pd.DataFrame:
    """Load hotspot rankings."""
    path = REPORTS_DIR / "hotspots.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_forecast_data() -> pd.DataFrame:
    """Load 2025-2030 forecast across all scenarios."""
    path = REPORTS_DIR / "forecast_2025_2030.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_metrics_data() -> Dict[str, Any]:
    """Load metrics JSON summary."""
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load model leaderboard CSV."""
    path = REPORTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_eda_summary() -> Dict[str, Any]:
    """Load EDA statistical facts."""
    path = REPORTS_DIR / "eda_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_trained_models() -> Dict[str, Any]:
    """Load serialized regression and classification models."""
    reg_path = MODELS_DIR / "final_regression_model.joblib"
    clf_path = MODELS_DIR / "final_classification_model.joblib"
    stats_path = MODELS_DIR / "training_stats.joblib"
    cols_path = MODELS_DIR / "feature_columns.json"

    models = {}
    if reg_path.exists():
        models["regression"] = joblib.load(reg_path)
    if clf_path.exists():
        models["classification"] = joblib.load(clf_path)
    if stats_path.exists():
        models["training_stats"] = joblib.load(stats_path)
    if cols_path.exists():
        with open(cols_path, "r", encoding="utf-8") as f:
            models["feature_columns"] = json.load(f)
    return models
