from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Mapping


def compact_int(n: int) -> str:
    """Return compact integer label used in output paths, e.g. 10000 -> 10k."""
    n = int(n)
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def compact_float(x: float) -> str:
    """Path-safe float label, e.g. 0.5 -> 0p5."""
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def auto_prepared_dir(
    dataset: str,
    n_items: int,
    n_blocks: int,
    top_k: int,
    embed_dim: int | None = None,
    root: str | Path = "data/processed",
) -> Path:
    """Default prepared-data path that adapts to the key experimental scale choices."""
    name = f"{dataset}_top{compact_int(n_items)}_b{int(n_blocks)}_k{int(top_k)}"
    if embed_dim is not None:
        name += f"_d{int(embed_dim)}"
    return Path(root) / name


def auto_sim_dir(
    prepared_dir: str | Path,
    p: float,
    mc: int,
    steps: int | None = None,
    root: str | Path = "outputs/sim",
) -> Path:
    """Default simulation-output path derived from prepared-data name and run settings."""
    base = Path(prepared_dir).name.rstrip("/") or "prepared"
    name = f"{base}_p{compact_float(p)}_mc{int(mc)}"
    if steps is not None:
        name += f"_steps{int(steps)}"
    return Path(root) / name


def write_json(path: str | Path, obj: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(obj), f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
