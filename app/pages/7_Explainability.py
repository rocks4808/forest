"""Page 7: Model Explainability & Feature Attribution."""

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
    top_nav_bar,
)

st.set_page_config(page_title="Explainability | CanopyIntel", page_icon="💡", layout="wide")
apply_custom_css()
top_nav_bar()

df_hotspots = load_hotspots_data()
df_hist = load_historical_data()

perm_path = ROOT_DIR / "reports" / "permutation_importance.csv"
df_perm = pd.read_csv(perm_path) if perm_path.exists() else pd.DataFrame()

insight_headline(
    "Explainable AI Audit: Autoregressive loss slopes, fire outbreaks, and agricultural expansion "
    "dominate predictive importance across the gradient boosted ensemble, while policy presence consistently exerts negative attribution."
)

tab_global, tab_local, tab_perm = st.tabs([
    "🌍 Global SHAP Attributions",
    "🎯 Local Regional Attributions",
    "🎲 Permutation Feature Importance",
])

# -------------------------------------------------------------------
# Tab 1: Global SHAP
# -------------------------------------------------------------------
with tab_global:
    st.markdown("##### Global Feature Impact on Deforestation Predictions")
    col_img1, col_img2 = st.columns(2)
    
    summary_fig = ROOT_DIR / "reports" / "figures" / "shap_summary.png"
    bar_fig = ROOT_DIR / "reports" / "figures" / "shap_bar.png"

    with col_img1:
        st.markdown("###### SHAP Summary (Beeswarm Distribution)")
        if summary_fig.exists():
            st.image(str(summary_fig), use_container_width=True)
            chart_caption("Each dot is a region-year. Red = high feature value, Blue = low. High fire_count strongly increases predicted loss.")
        else:
            st.info("Run `python -m src.train` to generate SHAP plots.")

    with col_img2:
        st.markdown("###### Mean Absolute Feature Attribution (|SHAP|)")
        if bar_fig.exists():
            st.image(str(bar_fig), use_container_width=True)
            chart_caption("Mean absolute impact ranking the top 15 most decisive predictors across all validation instances.")
        else:
            st.info("Run `python -m src.train` to generate SHAP plots.")

# -------------------------------------------------------------------
# Tab 2: Local Regional Attributions
# -------------------------------------------------------------------
with tab_local:
    st.markdown("##### Instance-Level Attribution & Plain-Language Explanation")
    
    all_regions = sorted(df_hist["region_id"].unique())
    sel_region = st.selectbox("Select Focal Region for Local Analysis", all_regions, index=0)

    # Top drivers from hotspot table
    hotspot_row = df_hotspots[df_hotspots["region_id"] == sel_region]
    if not hotspot_row.empty:
        drivers_str = hotspot_row["top_drivers"].iloc[0]
        pred_pct = hotspot_row["predicted_loss_pct_next"].iloc[0]
        risk_score = hotspot_row["risk_score"].iloc[0]
        tier = hotspot_row["tier"].iloc[0]

        st.markdown(
            f"""
            <div style="background-color: #16241D; border-left: 3px solid #7FB069; padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin-bottom: 1.5rem;">
                <div style="font-size: 1.1rem; font-family: 'Fraunces', serif; color: #E9E6DA; margin-bottom: 0.3rem;">
                    Region {sel_region} &mdash; Predicted 2025 Loss: <strong>{pred_pct:.2f}%</strong> (Risk Tier: {tier}, Score: {risk_score:.1f})
                </div>
                <div style="font-size: 0.85rem; color: #9FB0A4;">
                    Primary predictive drivers: <code>{drivers_str}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    waterfall_fig = ROOT_DIR / "reports" / "figures" / "shap_waterfall.png"
    if waterfall_fig.exists():
        st.image(str(waterfall_fig), use_container_width=True)
        chart_caption(f"Representative local waterfall plot illustrating push-and-pull contributions toward final prediction.")
    else:
        st.info("SHAP waterfall figure will appear here after training pipeline execution.")

# -------------------------------------------------------------------
# Tab 3: Permutation Importance
# -------------------------------------------------------------------
with tab_perm:
    st.markdown("##### Permutation Feature Importance (Validation Set)")
    if not df_perm.empty:
        fig_perm = px.bar(
            df_perm.head(15),
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            template=get_plotly_theme(),
            color="importance_mean",
            color_continuous_scale=["#121E18", "#7FB069"],
            labels={"importance_mean": "Mean Drop in MAE when Shuffled", "feature": "Engineered Feature"},
        )
        fig_perm.update_layout(height=450, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(t=20, b=30, l=150, r=20))
        st.plotly_chart(fig_perm, use_container_width=True)
        chart_caption("Permutation test shuffles feature values on unseen holdout records to measure true model reliance.")
    else:
        st.info("Run training to view permutation importance table.")

