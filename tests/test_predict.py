"""Unit tests for CanopyIntel CLI prediction module."""

import subprocess
import sys
import json
import pytest


def test_cli_predict_region():
    """Test single region CLI inference."""
    cmd = [sys.executable, "-m", "src.predict", "--region", "R01", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    assert len(data) == 1
    item = data[0]
    assert item["region_id"] == "R01"
    assert item["target_year"] == 2025
    assert 0.0 <= item["pred_loss_pct"] <= 100.0
    assert item["severity_tier"] in ["Low", "Medium", "High"]
    assert item["pred_loss_ha"] >= 0.0


def test_cli_predict_top():
    """Test top-N highest risk ranking via CLI."""
    top_n = 5
    cmd = [sys.executable, "-m", "src.predict", "--top", str(top_n), "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    assert len(data) == top_n
    # Verify sorted in descending order of predicted loss percentage
    loss_vals = [x["pred_loss_pct"] for x in data]
    assert loss_vals == sorted(loss_vals, reverse=True)
