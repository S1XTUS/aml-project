import os
from functools import lru_cache

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def project_path(*parts: str) -> str:
    """Resolve a path relative to the project root."""
    return os.path.join(PROJECT_ROOT, *parts)


def config_path(key: str) -> str:
    """Absolute path for an entry under `paths:` in config.yaml."""
    return project_path(load_config()["paths"][key])


def risk_level_from_score(score: float) -> str:
    """Single source of truth for LOW / MEDIUM / HIGH labelling."""
    thresholds = load_config()["thresholds"]
    if score >= thresholds["risk_high"]:
        return "HIGH"
    if score >= thresholds["risk_medium"]:
        return "MEDIUM"
    return "LOW"
