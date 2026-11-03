"""Page 3: Deforestation Hotspots & Degradation Analysis."""

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
    load_historical_data,
    load_hotspots_data,
    render_schematic_cartogram,
    top_nav_bar,
)

st.set_page_config(page_title="Hotspot Analysis | CanopyIntel", page_icon="🚨", layout="wide")
apply_custom_css()
top_nav_bar()

df_hotspots = load_hotspots_data()
df_hist = load_historical_data()

if df_hotspots.empty or df_hist.empty:
    st.warning("Hotspots or historical data not found. Please run `python -m src.train`.")
    st.stop()

# Computed headline
top_hotspot = df_hotspots.iloc[0]
n_critical = int((df_hotspots["tier"] == "Critical").sum())
n_accel = int((df_hotspots["trend_label"] == "Accelerating").sum())
insight_headline(
    f"Region {top_hotspot['region_id']} ranks as the #1 Critical hotspot (Risk Score {top_hotspot['risk_score']:.1f}/100, "
    f"predicted next-year loss {top_hotspot['predicted_loss_pct_next']:.2f}%). "
    f"Across all regions, {n_critical} are Critical and {n_accel} display accelerating 5-year degradation trends."
)

tab_rank, tab_heatmap, tab_scatter, tab_clusters = st.tabs([
    "🏆 Risk Leaderboard",
    "🗺️ Region × Year Heatmap",
    "🫧 Risk vs. Scale Matrix",
    "🧩 Structural Archetypes",
])

# -------------------------------------------------------------------
# Tab 1: Ranked Hotspots Table
# -------------------------------------------------------------------
with tab_rank:
    st.markdown("##### Prioritized Deforestation Hotspots")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        tier_filter = st.multiselect("Filter Tiers", ["Critical", "High", "Moderate", "Low"], default=["Critical", "High", "Moderate", "Low"])
    with col_ctrl2:
        search_query = st.text_input("Search Region or Driver", "")

    display_df = df_hotspots[df_hotspots["tier"].isin(tier_filter)].copy()
    if search_query:
        display_df = display_df[
            display_df["region_id"].str.contains(search_query, case=False)
            | display_df["top_drivers"].str.contains(search_query, case=False)
            | display_df["cluster_archetype"].str.contains(search_query, case=False)
        ]

    # Add trend arrows
    trend_map = {"Accelerating": "▲ Accelerating", "Improving": "▼ Improving", "Stable": "▬ Stable"}
    display_df["trend_display"] = display_df["trend_label"].map(trend_map)

    st.dataframe(
        display_df[[
            "region_id", "risk_score", "tier", "trend_display", "predicted_loss_pct_next",
            "cluster_archetype", "top_drivers", "forest_area_ha"
        ]],
        use_container_width=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "predicted_loss_pct_next": st.column_config.NumberColumn("Pred Loss (2025)", format="%.2f%%"),
            "forest_area_ha": st.column_config.NumberColumn("Forest Area (ha)", format="%d"),
            "trend_display": st.column_config.TextColumn("5-Yr Trend"),
            "cluster_archetype": st.column_config.TextColumn("Archetype"),
            "top_drivers": st.column_config.TextColumn("Top SHAP Drivers"),
        },
        hide_index=True,
    )
    st.caption("Risk score synthesizes loss forecast (35%), 5-yr slope (20%), high-severity prob (20%), fire pressure (10%), human pressure (10%), and conservation deficit (5%).")
    
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Hotspots CSV", csv_bytes, "hotspots.csv", "text/csv")

# -------------------------------------------------------------------
# Tab 2: Region x Year Heatmap
# -------------------------------------------------------------------
with tab_heatmap:
    st.markdown("##### Historical Loss Intensity (Sorted by 2025 Risk)")
    # Pivot matrix: rows = region (sorted by risk_score), columns = year, values = loss_pct
    sorted_regions = df_hotspots["region_id"].tolist()
    pivot_df = df_hist.pivot(index="region_id", columns="year", values="loss_pct")
    pivot_df = pivot_df.reindex(sorted_regions)

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(y) for y in pivot_df.columns],
        y=pivot_df.index.tolist(),
        colorscale=[[0, "#121E18"], [0.35, "#7FB069"], [0.65, "#E07A3F"], [1, "#C8402F"]],
        zmin=0.0,
        zmax=2.8,
        colorbar=dict(title="Loss %", tickfont=dict(family="IBM Plex Mono")),
    ))
    fig_heat.update_layout(
        template=get_plotly_theme(),
        height=750,
        margin=dict(l=60, r=30, t=20, b=40),
        xaxis_title="Year",
        yaxis_title="Region (Ranked Top Risk → Lowest)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    chart_caption("Notice the distinct structural divide between top-ranked regions (chronic multi-year loss rates >2.0%) and bottom-ranked protected areas.")

# -------------------------------------------------------------------
# Tab 3: Risk vs Scale Bubble Matrix
# -------------------------------------------------------------------
with tab_scatter:
    st.markdown("##### Risk Score vs. Total Standing Forest Area")
    fig_scatter = px.scatter(
        df_hotspots,
        x="forest_area_ha",
        y="risk_score",
        size="predicted_loss_pct_next",
        color="tier",
        hover_name="region_id",
        hover_data={"forest_area_ha": ":,", "predicted_loss_pct_next": ":.2f", "cluster_archetype": True},
        color_discrete_map={"Critical": "#C8402F", "High": "#E07A3F", "Moderate": "#D9B84A", "Low": "#7FB069"},
        template=get_plotly_theme(),
        labels={"forest_area_ha": "Forest Area (Hectares)", "risk_score": "Composite Risk Score (0-100)"},
    )
    fig_scatter.update_layout(height=450, margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(fig_scatter, use_container_width=True)
    chart_caption("High-leverage intervention targets reside in the upper-right quadrant: large standing forest reserves experiencing severe degradation pressures.")

# -------------------------------------------------------------------
# Tab 4: Regional Archetypes (Clustering)
# -------------------------------------------------------------------
with tab_clusters:
    st.markdown("##### Regional Typologies (K-Means Structural Clustering)")
    arch_summary = df_hotspots.groupby("cluster_archetype").agg({
        "region_id": "count",
        "risk_score": "mean",
        "predicted_loss_pct_next": "mean",
        "forest_area_ha": "sum",
    }).reset_index().rename(columns={"region_id": "Regions Count", "risk_score": "Avg Risk Score", "predicted_loss_pct_next": "Avg Pred Loss %", "forest_area_ha": "Total Area (ha)"})

    st.dataframe(
        arch_summary,
        use_container_width=True,
        column_config={
            "Avg Risk Score": st.column_config.NumberColumn(format="%.1f"),
            "Avg Pred Loss %": st.column_config.NumberColumn(format="%.2f%%"),
            "Total Area (ha)": st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
    )
    chart_caption("Four structural archetypes identified based on multivariate profiles of fire, agricultural expansion, road density, and historical loss.")

