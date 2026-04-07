from __future__ import annotations

"""
Config loader for volatility-regime-engine.

Loads config.yaml once at startup and provides it as a plain dict
to all downstream modules. No module should read config.yaml directly.

Financial rationale: centralizing parameters ensures backtests are
reproducible and parameter changes are auditable in a single file.
"""

import os
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """
    Load configuration from config.yaml and return as a nested dict.

    Parameters
    ----------
    config_path : str or None
        Absolute or relative path to config.yaml.
        If None, defaults to config.yaml in the project root.

    Returns
    -------
    dict
        Full configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist at the resolved path.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
