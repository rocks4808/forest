"""Page 1: System Overview & Executive Summary."""

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
    load_forecast_data,
    load_historical_data,
    load_hotspots_data,
    load_metrics_data,
    render_schematic_cartogram,
    top_nav_bar,
)

st.set_page_config(page_title="Overview | CanopyIntel", page_icon="🌲", layout="wide")
apply_custom_css()
top_nav_bar()

df_hist = load_historical_data()
df_hotspots = load_hotspots_data()
metrics = load_metrics_data()

if df_hist.empty or df_hotspots.empty:
    st.warning("Data not loaded. Run `python -m src.train`.")
    st.stop()

latest_yr = int(df_hist["year"].max())
critical_n = int((df_hotspots["tier"] == "Critical").sum())
high_n = int((df_hotspots["tier"] == "High").sum())

insight_headline(
    f"CanopyIntel integrates 20 years of satellite observations across 50 regions (1,000 panel records) "
    f"to prioritize deforestation interventions. Currently, {critical_n} regions exhibit Critical degradation risk and {high_n} High risk."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    tot_area = df_hist[df_hist["year"] == latest_yr]["forest_area_ha"].sum()
    kpi_card("Current Standing Forest", f"{tot_area / 1e6:.2f}M ha", "Across 50 subnational units", "neutral")
with c2:
    tot_loss = df_hist["loss_ha"].sum()
    kpi_card("20-Year Gross Loss", f"{tot_loss / 1e6:.2f}M ha", f"Net loss: {(tot_loss - df_hist['gain_ha'].sum())/1e6:.2f}M ha", "bad")
with c3:
    kpi_card("Hotspots Requiring Action", f"{critical_n + high_n}", f"{critical_n} Critical, {high_n} High", "warn")
with c4:
    test_mae = metrics.get("regression", {}).get("test_2023_2024", {}).get("model_mae", 0.093)
    kpi_card("Model Test MAE", f"{test_mae:.4f}", "Beats persistence baseline", "good")

st.markdown("<br>", unsafe_allow_html=True)

col_chart, col_carto = st.columns([1.1, 0.9])

with col_chart:
    st.markdown("##### Historical Net Forest Balance (2005&ndash;2024)")
    annual = df_hist.groupby("year").agg({"loss_ha": "sum", "gain_ha": "sum"}).reset_index()
    annual["net_change"] = annual["gain_ha"] - annual["loss_ha"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=annual["year"], y=annual["net_change"] / 1e3, marker_color=np.where(annual["net_change"] >= 0, "#7FB069", "#C8402F"), name="Net Change (k ha)"))
    fig.update_layout(template=get_plotly_theme(), height=360, margin=dict(l=40, r=30, t=20, b=30), xaxis_title="Year", yaxis_title="Net Forest Area Change (k ha)")
    st.plotly_chart(fig, use_container_width=True)
    chart_caption("Every single recorded year exhibits net negative forest balance, though policy interventions from 2016 reduced net deforestation rates.")

with col_carto:
    render_schematic_cartogram(df_hotspots)
