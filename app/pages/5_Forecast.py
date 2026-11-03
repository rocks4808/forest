"""Page 5: Multi-Step Deforestation Forecast to 2030."""

from __future__ import annotations

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
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
    top_nav_bar,
)

st.set_page_config(page_title="2025-2030 Forecast | CanopyIntel", page_icon="🔮", layout="wide")
apply_custom_css()
top_nav_bar()

df_hist = load_historical_data()
df_forecast = load_forecast_data()

backtest_path = ROOT_DIR / "reports" / "forecast_backtest.csv"
df_backtest = pd.read_csv(backtest_path) if backtest_path.exists() else pd.DataFrame()

if df_forecast.empty or df_hist.empty:
    st.warning("Forecast artifacts not found. Please run `python -m src.train`.")
    st.stop()

# Computed headline
base_2030_loss = df_forecast[(df_forecast["year"] == 2030) & (df_forecast["scenario"] == "baseline")]["loss_ha_pred"].sum()
policy_2030_loss = df_forecast[(df_forecast["year"] == 2030) & (df_forecast["scenario"] == "policy_on_everywhere")]["loss_ha_pred"].sum()
ha_saved = base_2030_loss - policy_2030_loss

insight_headline(
    f"Under the baseline scenario, 50 regions are projected to lose a cumulative {df_forecast[df_forecast['scenario'] == 'baseline']['loss_ha_pred'].sum() / 1e6:.2f}M hectares by 2030. "
    f"Universal policy activation could avert approximately {ha_saved:,.0f} hectares of loss in 2030 alone."
)

col_sc1, col_sc2 = st.columns([1, 1.5])
with col_sc1:
    scenario_sel = st.selectbox(
        "Select Policy & Climate Scenario",
        ["baseline", "policy_on_everywhere", "conservation_push", "high_fire", "agri_surge"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
with col_sc2:
    all_regs = sorted(df_hist["region_id"].unique())
    region_sel = st.selectbox("Select Focal Region", ["National Aggregate (All Regions)"] + all_regs)

tab_fc, tab_area, tab_top10, tab_backtest = st.tabs([
    "📈 Loss Rate Trajectory & Intervals",
    "🌲 Standing Forest Area (Mass Balance)",
    "🏆 Top 10 Cumulative Loss Hotspots",
    "🔍 Historical Backtest Verification",
])

# -------------------------------------------------------------------
# Tab 1: Loss Rate Trajectory
# -------------------------------------------------------------------
with tab_fc:
    st.markdown("##### Historical Record & Recursive Forecast to 2030")

    fig_fc = go.Figure()
    if region_sel == "National Aggregate (All Regions)":
        # Aggregate historical
        hist_agg = df_hist.groupby("year")["loss_pct"].mean().reset_index()
        fc_sub = df_forecast[df_forecast["scenario"] == scenario_sel].groupby("year").agg({
            "loss_pct_pred": "mean",
            "loss_pct_lo": "mean",
            "loss_pct_hi": "mean",
        }).reset_index()

        # Connect 2024 to 2025
        last_hist = hist_agg.iloc[-1]
        bridge = pd.DataFrame([{
            "year": last_hist["year"],
            "loss_pct_pred": last_hist["loss_pct"],
            "loss_pct_lo": last_hist["loss_pct"],
            "loss_pct_hi": last_hist["loss_pct"],
        }])
        fc_plot = pd.concat([bridge, fc_sub], ignore_index=True)

        fig_fc.add_trace(go.Scatter(x=hist_agg["year"], y=hist_agg["loss_pct"], mode="lines+markers", name="Historical Mean (2005-2024)", line=dict(color="#E9E6DA", width=2.5)))
        # Shaded 80% interval
        fig_fc.add_trace(go.Scatter(x=fc_plot["year"], y=fc_plot["loss_pct_hi"], mode="lines", line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=fc_plot["year"], y=fc_plot["loss_pct_lo"], mode="lines", fill="tonexty", fillcolor="rgba(127, 176, 105, 0.2)", line=dict(width=0), name="80% Prediction Interval"))
        fig_fc.add_trace(go.Scatter(x=fc_plot["year"], y=fc_plot["loss_pct_pred"], mode="lines+markers", name=f"Forecast: {scenario_sel.replace('_', ' ').title()}", line=dict(color="#7FB069", width=2.5, dash="dash")))

    else:
        # Regional trajectory
        r_hist = df_hist[df_hist["region_id"] == region_sel].sort_values("year")
        r_fc = df_forecast[(df_forecast["region_id"] == region_sel) & (df_forecast["scenario"] == scenario_sel)].sort_values("year")
        
        last_h = r_hist.iloc[-1]
        bridge = pd.DataFrame([{
            "year": last_h["year"],
            "loss_pct_pred": last_h["loss_pct"],
            "loss_pct_lo": last_h["loss_pct"],
            "loss_pct_hi": last_h["loss_pct"],
        }])
        fc_plot = pd.concat([bridge, r_fc], ignore_index=True)

        fig_fc.add_trace(go.Scatter(x=r_hist["year"], y=r_hist["loss_pct"], mode="lines+markers", name=f"Historical {region_sel}", line=dict(color="#E9E6DA", width=2.5)))
        fig_fc.add_trace(go.Scatter(x=fc_plot["year"], y=fc_plot["loss_pct_hi"], mode="lines", line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=fc_plot["year"], y=fc_plot["loss_pct_lo"], mode="lines", fill="tonexty", fillcolor="rgba(127, 176, 105, 0.25)", line=dict(width=0), name="80% Uncertainty Band"))
        fig_fc.add_trace(go.Scatter(x=fc_plot["year"], y=fc_plot["loss_pct_pred"], mode="lines+markers", name=f"Projected ({scenario_sel})", line=dict(color="#7FB069", width=2.5, dash="dash")))

    # Labeled divider at 2024 forecast boundary
    fig_fc.add_vline(x=2024, line_width=1.5, line_dash="dot", line_color="#E07A3F")
    fig_fc.add_annotation(x=2024, y=1.0, yref="paper", text="Forecast Origin (2024)", showarrow=False, font=dict(family="IBM Plex Mono", color="#E07A3F", size=11), xanchor="right", yanchor="top")

    fig_fc.update_layout(
        template=get_plotly_theme(),
        height=420,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Year",
        yaxis_title="Annual Loss % of Forest Area",
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    chart_caption("Multi-step autoregressive roll-forward with dynamic feature recalculation and calibrated uncertainty bands.")

# -------------------------------------------------------------------
# Tab 2: Standing Forest Area (Mass Balance)
# -------------------------------------------------------------------
with tab_area:
    st.markdown("##### Forest Area Projection Under Physical Mass Balance: A(t+1) = A(t) - Loss + Gain")
    if region_sel == "National Aggregate (All Regions)":
        area_hist = df_hist.groupby("year")["forest_area_ha"].sum() / 1e6
        area_fc = df_forecast[df_forecast["scenario"] == scenario_sel].groupby("year")["forest_area_pred"].sum() / 1e6
    else:
        area_hist = df_hist[df_hist["region_id"] == region_sel].set_index("year")["forest_area_ha"] / 1e6
        area_fc = df_forecast[(df_forecast["region_id"] == region_sel) & (df_forecast["scenario"] == scenario_sel)].set_index("year")["forest_area_pred"] / 1e6

    full_series = pd.concat([area_hist, area_fc])
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(x=area_hist.index, y=area_hist.values, mode="lines+markers", name="Historical Area", line=dict(color="#7FB069", width=2.5)))
    fig_area.add_trace(go.Scatter(x=[area_hist.index[-1]] + list(area_fc.index), y=[area_hist.values[-1]] + list(area_fc.values), mode="lines+markers", name="Projected Area", line=dict(color="#D9B84A", width=2.5, dash="dash")))
    fig_area.add_vline(x=2024, line_width=1.5, line_dash="dot", line_color="#E07A3F")

    fig_area.update_layout(
        template=get_plotly_theme(),
        height=380,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Year",
        yaxis_title="Standing Forest (Million Hectares)",
    )
    st.plotly_chart(fig_area, use_container_width=True)
    chart_caption("Physical mass balance guarantees that forest area cannot go negative and strictly satisfies inventory conservation.")

# -------------------------------------------------------------------
# Tab 3: Top 10 Cumulative Loss
# -------------------------------------------------------------------
with tab_top10:
    st.markdown("##### Top 10 Regions by Projected Cumulative Forest Loss (2025&ndash;2030)")
    top10 = (
        df_forecast[df_forecast["scenario"] == scenario_sel]
        .groupby("region_id")["loss_ha_pred"]
        .sum()
        .reset_index()
        .sort_values("loss_ha_pred", ascending=False)
        .head(10)
    )
    fig_top10 = px.bar(
        top10,
        x="region_id",
        y="loss_ha_pred",
        color="loss_ha_pred",
        color_continuous_scale=["#D9B84A", "#C8402F"],
        template=get_plotly_theme(),
        labels={"region_id": "Region", "loss_ha_pred": "Cumulative Projected Loss (Hectares)"},
    )
    fig_top10.update_layout(height=350, margin=dict(l=30, r=30, t=20, b=40), coloraxis_showscale=False)
    st.plotly_chart(fig_top10, use_container_width=True)

# -------------------------------------------------------------------
# Tab 4: Historical Backtest Verification
# -------------------------------------------------------------------
with tab_backtest:
    st.markdown("##### Forecast Method Backtest (Simulated from 2019 & 2021 Origins)")
    if not df_backtest.empty:
        st.dataframe(
            df_backtest,
            use_container_width=True,
            column_config={
                "MAE_loss_pct": st.column_config.NumberColumn("Horizon MAE (%)", format="%.4f%%"),
                "horizon_step": st.column_config.NumberColumn("Forecast Step (Years Ahead)"),
            },
            hide_index=True,
        )
        chart_caption("As expected in recursive panel forecasting, projection error scales proportionally with horizon distance h.")
    else:
        st.info("Run full training pipeline to view multi-horizon backtest table.")

