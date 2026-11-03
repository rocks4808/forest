"""Streamlit App integration tests using Streamlit AppTest framework."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent

PAGE_FILES = [
    "app/streamlit_app.py",
    "app/pages/1_Overview.py",
    "app/pages/2_Data_Explorer.py",
    "app/pages/3_Hotspot_Analysis.py",
    "app/pages/4_Model_Performance.py",
    "app/pages/5_Forecast.py",
    "app/pages/6_What_If_Simulator.py",
    "app/pages/7_Explainability.py",
    "app/pages/8_Executive_Brief.py",
]

@pytest.mark.parametrize("page_file", PAGE_FILES)
def test_streamlit_pages_load_cleanly(page_file: str):
    """Verify that every Streamlit page compiles and runs without exceptions."""
    full_path = ROOT_DIR / page_file
    assert full_path.exists(), f"Page file {full_path} does not exist"
    at = AppTest.from_file(str(full_path), default_timeout=30)
    at.run()
    # Ensure there are no unhandled exceptions raised during execution
    assert len(at.exception) == 0, f"Page {page_file} encountered exceptions: {[e.value for e in at.exception]}"


def test_sidebar_render_and_components():
    """Verify that render_sidebar and custom CSS integrate smoothly without error."""
    from app.components import render_sidebar, load_hotspots_data, load_historical_data
    import pandas as pd

    df_hist = load_historical_data()
    df_hotspots = load_hotspots_data()
    assert not df_hist.empty
    assert not df_hotspots.empty

    # Test with AppTest running main app to verify sidebar contains our custom widgets
    at = AppTest.from_file(str(ROOT_DIR / "app/streamlit_app.py"), default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    # Verify sidebar selectbox exists
    assert len(at.sidebar.selectbox) >= 1
    assert "R44" in at.sidebar.selectbox[0].options
