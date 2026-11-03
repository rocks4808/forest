"""Page 6: Interactive What-If Policy & Climate Simulator."""

from __future__ import annotations

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import (
    apply_custom_css,
    chart_caption,
    get_plotly_theme,
    insight_headline,
    kpi_card,
    load_historical_data,
    load_hotspots_data,
    load_metrics_data,
    load_trained_models,
    tier_badge,
    top_nav_bar,
)
from src.features import build_features

st.set_page_config(page_title="What-If Simulator | CanopyIntel", page_icon="🎛️", layout="wide")
apply_custom_css()
top_nav_bar()

df_hist = load_historical_data()
df_hotspots = load_hotspots_data()
models = load_trained_models()
metrics = load_metrics_data()

if df_hist.empty or "regression" not in models:
    st.warning("Trained model or historical data not loaded. Please run `python -m src.train`.")
    st.stop()

insight_headline(
    "Interactive What-If Simulation Console: Test sensitivity to conservation policies, agricultural frontiers, "
    "and climate shock scenarios in real time using cached model inference."
)

st.markdown(
    """
    > [!NOTE]
    > **Disclaimer:** Estimates are produced by empirical ML models trained on synthetic historical panel data.
    > Relationships represent statistical associations rather than proven causal mechanisms.
    """
)

all_regions = sorted(df_hist["region_id"].unique())

# Split into Control Rail (left) and Results Panel (right)
ctrl_col, res_col = st.columns([1, 1.25])

with ctrl_col:
    st.markdown("##### 🎛️ Regional Levers & Policy Switches")
    focal_region = st.selectbox("Select Target Region", all_regions, index=0)
    
    # Preset selection
    preset = st.selectbox(
        "One-Click Scenario Preset",
        ["Custom Manual Settings", "Baseline (No Change)", "Policy On Everywhere", "High Fire Shock (+25%)", "Agri Surge (+30%)", "Conservation Push (Policy On & Fire -20%)"]
    )

    # Defaults from 2024
    reg_history = df_hist[df_hist["region_id"] == focal_region].sort_values("year")
    latest_row = reg_history.iloc[-1]

    # Preset logic
    def_fire_delta = 0
    def_agri_delta = 0
    def_pop_delta = 0
    def_policy = bool(latest_row["policy_active"] == 1)
    def_protected = bool(latest_row["protected_area"] == 1)

    if preset == "Policy On Everywhere":
        def_policy = True
    elif preset == "High Fire Shock (+25%)":
        def_fire_delta = 25
    elif preset == "Agri Surge (+30%)":
        def_agri_delta = 30
    elif preset == "Conservation Push (Policy On & Fire -20%)":
        def_policy = True
        def_fire_delta = -20

    fire_delta = st.slider("Fire Activity Delta (%)", -50, 100, def_fire_delta, step=5, help="Simulate increase or decrease in fire events")
    agri_delta = st.slider("Agricultural Expansion Delta (%)", -50, 100, def_agri_delta, step=5)
    pop_delta = st.slider("Population Growth Delta (%)", -20, 50, def_pop_delta, step=5)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        policy_toggle = st.toggle("Conservation Policy Active", value=def_policy)
    with col_t2:
        protected_toggle = st.toggle("Protected Area Status", value=def_protected)

# -------------------------------------------------------------------
# Model Inference for What-If
# -------------------------------------------------------------------
with res_col:
    st.markdown("##### 🔮 Immediate Forecast Response (Year 2025)")

    # Baseline 2025 from hotspots
    hotspot_row = df_hotspots[df_hotspots["region_id"] == focal_region]
    base_pred_pct = float(hotspot_row["predicted_loss_pct_next"].iloc[0]) if not hotspot_row.empty else float(latest_row["loss_pct"])
    base_area = int(latest_row["forest_area_ha"])

    # Simulate perturbed row
    sim_row = latest_row.copy()
    sim_row["fire_count"] = max(0, int(round(sim_row["fire_count"] * (1 + fire_delta / 100.0))))
    sim_row["agri_expansion_pct"] = max(0.0, float(sim_row["agri_expansion_pct"] * (1 + agri_delta / 100.0)))
    sim_row["population_density"] = max(0.0, float(sim_row["population_density"] * (1 + pop_delta / 100.0)))
    sim_row["policy_active"] = 1 if policy_toggle else 0
    sim_row["protected_area"] = 1 if protected_toggle else 0

    # Build features on simulated frame
    sim_history = reg_history.copy()
    sim_history.iloc[-1] = sim_row
    feat_df, _ = build_features(sim_history, training_stats=models.get("training_stats"))
    
    feature_cols = models.get("feature_columns", [])
    X_sim = feat_df.iloc[[-1]][feature_cols].fillna(0.0)

    # Predict
    reg_model = models["regression"]
    raw_p = float(reg_model.predict(X_sim)[0])

    target_form = metrics.get("metadata", {}).get("target_form", "log_ratio")
    curr_l = float(latest_row["loss_pct"])
    if target_form == "delta":
        sim_pred_pct = curr_l + raw_p
    elif target_form == "log_ratio":
        sim_pred_pct = curr_l * np.exp(raw_p)
    else:
        sim_pred_pct = raw_p

    sim_pred_pct = float(np.clip(sim_pred_pct, 0.01, 15.0))
    delta_loss = sim_pred_pct - base_pred_pct

    # KPI Summary
    k1, k2, k3 = st.columns(3)
    with k1:
        d_type = "good" if delta_loss < -0.01 else ("bad" if delta_loss > 0.01 else "neutral")
        kpi_card("Simulated Loss Rate", f"{sim_pred_pct:.2f}%", f"{delta_loss:+.2f}% vs Baseline", d_type)
    with k2:
        sim_loss_ha = int(round(sim_pred_pct * base_area / 100.0))
        base_loss_ha = int(round(base_pred_pct * base_area / 100.0))
        diff_ha = sim_loss_ha - base_loss_ha
        d_type_ha = "good" if diff_ha < 0 else ("bad" if diff_ha > 0 else "neutral")
        kpi_card("Projected Gross Loss", f"{sim_loss_ha:,} ha", f"{diff_ha:+,} ha delta", d_type_ha)
    with k3:
        # Severity
        thresh = metrics.get("thresholds", {"low_upper": 0.97, "high_lower": 1.46})
        if sim_pred_pct <= thresh["low_upper"]:
            sev = "Low"
            color = "#7FB069"
        elif sim_pred_pct <= thresh["high_lower"]:
            sev = "Medium"
            color = "#D9B84A"
        else:
            sev = "High"
            color = "#C8402F"
        kpi_card("Degradation Severity", sev, f"Thresholds: <{thresh['low_upper']:.2f}% | >{thresh['high_lower']:.2f}%", "good" if sev == "Low" else "warn")

    # Side-by-side comparison bar
    st.markdown("<br>", unsafe_allow_html=True)
    fig_comp = go.Figure(data=[
        go.Bar(name="Baseline Setting", x=["Loss Rate (%)"], y=[base_pred_pct], marker_color="#6B7F73", text=[f"{base_pred_pct:.2f}%"], textposition="auto"),
        go.Bar(name="Simulated Scenario", x=["Loss Rate (%)"], y=[sim_pred_pct], marker_color=color, text=[f"{sim_pred_pct:.2f}%"], textposition="auto"),
    ])
    fig_comp.update_layout(
        template=get_plotly_theme(),
        height=280,
        barmode="group",
        yaxis_title="Annual Loss Rate (%)",
        margin=dict(t=20, b=30, l=40, r=20),
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    chart_caption("Instant inference updates dynamically as sliders change.")

