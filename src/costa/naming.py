from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def format_count(n: int) -> str:
    """Compact count for directory names: 10000 -> 10k, 15000 -> 15k, 12500 -> 12p5k."""
    n = int(n)
    if n >= 1000:
        val = n / 1000.0
        if abs(val - round(val)) < 1e-12:
            return f"{int(round(val))}k"
        return f"{val:.1f}".rstrip("0").rstrip(".").replace(".", "p") + "k"
    return str(n)


def format_probability(p: float) -> str:
    """Filesystem-safe probability label: 0.5 -> 0p5, 0.25 -> 0p25."""
    s = f"{float(p):.4g}"
    return s.replace("-", "m").replace(".", "p")


def prepared_dir_name(dataset: str, n_items: int, n_blocks: int, top_k: int, embed_dim: int | None = None) -> str:
    """Automatic prepared-data stem adapting to dataset, top-N item count, B, and top-K.

    The default stem uses the ``top{N}`` convention because preprocessing keeps
    the top-N items by weighted interaction volume. If ``embed_dim`` is supplied,
    append it for cache disambiguation.
    """
    stem = f"{dataset}_top{format_count(n_items)}_b{int(n_blocks)}_k{int(top_k)}"
    if embed_dim is not None:
        stem += f"_d{int(embed_dim)}"
    return stem


def _design_label(designs: str | Iterable[str] | None) -> str | None:
    if designs is None:
        return None
    if isinstance(designs, str):
        parts = [x.strip() for x in designs.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in designs if str(x).strip()]
    if not parts:
        return None
    default = ["independent", "rbsd_approx", "cluster_leiden", "static_graph_ocd", "COSTA", "COSTA_cluster_no_kro", "COSTA_cluster_kro", "COSTA_IC_lam0.2", "COSTA_IC_lam1", "COSTA_IC_lam5"]
    if parts == default or [p.lower() for p in parts] == [p.lower() for p in default]:
        return None
    canon = {
        "independent": "ind",
        "rbsd_approx": "rbsd",
        "cluster_leiden": "leid",
        "static_graph_ocd": "ocd",
        "COSTA": "costa",
        "costa": "costa",
        "COSTA_cluster_no_kro": "ccnokro",
        "costa_cluster_no_kro": "ccnokro",
        "COSTA_cluster_kro": "cckro",
        "costa_cluster_kro": "cckro",
        "COSTA_IC_lam0.2": "iclam0p2",
        "costa_ic_lam0.2": "iclam0p2",
        "COSTA_IC_lam0p2": "iclam0p2",
        "costa_ic_lam0p2": "iclam0p2",
        "COSTA_IC_lam1": "iclam1",
        "costa_ic_lam1": "iclam1",
        "COSTA_IC_lam5": "iclam5",
        "costa_ic_lam5": "iclam5",
    }
    labels = [canon.get(x, canon.get(x.lower(), x.replace("_", "").replace(".", "p"))) for x in parts]
    if len(labels) == 1:
        return labels[0]
    return "designs" + "-".join(labels)

def simulation_dir_name(
    prepared_dir: str | Path,
    p: float,
    mc: int,
    steps: int | None = None,
    seed: int | None = None,  # deprecated; kept for backward-compatible calls
    d_item: int | None = None,
    d_time: int | None = None,
    designs: str | Iterable[str] | None = None,
) -> str:
    """Automatic simulation-output stem.

    Kronecker dimensions are included when supplied. This prevents COSTA
    dimension-ablation runs with d_item=16,32,64 from overwriting one another.
    """
    stem = f"{Path(prepared_dir).name}_p{format_probability(p)}_mc{int(mc)}"
    if steps is not None:
        stem += f"_steps{int(steps)}"
    if d_item is not None:
        stem += f"_di{int(d_item)}"
    if d_time is not None:
        stem += f"_dt{int(d_time)}"
    tag = _design_label(designs)
    if tag:
        stem += f"_{tag}"
    # The base seed is intentionally not included in the auto-generated directory
    # name. It is stored in run_config.json and simulation_draws.csv instead.
    # This keeps paired-design output directories cleaner, e.g. no trailing
    # `_seed123` when all methods share the same base seed.
    return stem

def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, allow_nan=True)


def read_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else dict(default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
