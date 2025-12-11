# src/utils/config.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """
    Simple wrapper around a config dictionary.

    You can access the raw dictionary via .data.
    """

    data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        # Return a deep copy to avoid accidental in-place modifications
        return copy.deepcopy(self.data)


def load_config(
    path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load a YAML config file and optionally override some fields.

    Parameters
    ----------
    path : str
        Path to the YAML file.
    overrides : Optional[Dict[str, Any]]
        Dictionary with values that should override config entries.
        Supports only top-level keys for simplicity.

    Returns
    -------
    Config
        A simple configuration object.
    """
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    if raw_cfg is None:
        raw_cfg = {}

    if overrides:
        for key, value in overrides.items():
            raw_cfg[key] = value

    return Config(raw_cfg)
