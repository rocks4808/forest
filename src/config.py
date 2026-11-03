"""Configuration loader for the Forest Loss ML pipeline."""
import os
import yaml
from typing import Any, Dict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if path is None:
        path = os.path.join(ROOT_DIR, "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

SEED = 42

