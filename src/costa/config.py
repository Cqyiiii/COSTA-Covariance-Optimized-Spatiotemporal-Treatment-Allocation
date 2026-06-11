from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _read_yaml_raw(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install requirements.txt.")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return dict(data)


def read_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Read a YAML config, supporting a lightweight ``extends`` key.

    ``extends: base.yaml`` is resolved relative to the config file's directory.
    Nested dictionaries are merged recursively, with the child overriding the
    parent. This keeps RetailRocket and MovieLens profiles on one shared schema.
    """
    if path is None:
        return {}
    p = Path(path)
    data = _read_yaml_raw(p)
    parent = data.get("extends")
    if parent:
        parents = parent if isinstance(parent, list) else [parent]
        merged: dict[str, Any] = {}
        for par in parents:
            par_path = Path(par)
            if not par_path.is_absolute():
                par_path = p.parent / par_path
            merged = _deep_update(merged, read_yaml_config(par_path))
        return _deep_update(merged, data)
    return data


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config sections into argparse-style defaults.

    Section order matters: data defaults are applied first, then simulation,
    outcomes, optimization, and monitoring. Hence simulation.output_root wins over
    data.output_root when both are present, as intended for run_simulation.py.
    """
    out: dict[str, Any] = {}
    section_order = ["data", "simulation", "outcome", "outcomes", "optimization", "monitoring"]
    for k, v in cfg.items():
        if isinstance(v, dict) and k not in section_order:
            continue
        if k not in section_order and k != "extends":
            out[k] = v
    for sec in section_order:
        v = cfg.get(sec)
        if isinstance(v, dict):
            out.update(v)
    return out


def set_argparse_defaults_from_config(parser, config_path: str | Path | None) -> dict[str, Any]:
    cfg = flatten_config(read_yaml_config(config_path))
    # argparse uses underscores, while YAML may use hyphens.
    normalized = {str(k).replace("-", "_"): v for k, v in cfg.items()}
    if isinstance(normalized.get("models"), list):
        normalized["models"] = ",".join(str(x) for x in normalized["models"])
    if isinstance(normalized.get("designs"), list):
        normalized["designs"] = ",".join(str(x) for x in normalized["designs"])
    if normalized:
        parser.set_defaults(**normalized)
    return normalized
