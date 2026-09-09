from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root_from_config(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(project_root_from_config(config_path))
    return cfg


def resolve_path(cfg: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

