from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .naming import prepared_dir_name


@dataclass(frozen=True)
class SimFolder:
    """Parsed metadata for an existing simulation output folder."""

    out_dir: Path
    dataset: str
    n_items: int
    n_blocks: int
    top_k: int
    p: float
    mc: int
    steps: int
    d_item: int
    d_time: int
    suffix: str = ""
    run_config: dict[str, Any] | None = None

    @property
    def dataset_key(self) -> str:
        ds = self.dataset.lower()
        if "retailrocket" in ds:
            return "retailrocket"
        if "movielens" in ds:
            return "movielens"
        return ds


def read_json_safely(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def parse_count_token(tok: str) -> int:
    tok = str(tok).lower().strip()
    if tok.endswith("k"):
        return int(float(tok[:-1].replace("p", ".")) * 1000)
    return int(tok)


def parse_p_token(tok: str) -> float:
    return float(str(tok).replace("p", "."))


_SIM_RE = re.compile(
    r"^(?P<dataset>.+?)_top(?P<n_items>[0-9]+(?:p[0-9]+)?k|[0-9]+)"
    r"_b(?P<B>[0-9]+)_k(?P<top_k>[0-9]+)"
    r"_p(?P<p>[0-9]+p[0-9]+)_mc(?P<mc>[0-9]+)"
    r"_steps(?P<steps>[0-9]+)_di(?P<d_item>[0-9]+)_dt(?P<d_time>[0-9]+)"
    r"(?P<suffix>.*)$"
)


def parse_sim_folder_name(name: str) -> dict[str, Any]:
    """Parse our simulation folder naming convention.

    Examples
    --------
    retailrocket_top10k_b8_k10_p0p5_mc500_steps400_di16_dt7
    movielens_top10k_b8_k10_p0p1_mc500_steps400_di16_dt7_designsind-rbsd-leid-costa-ccnokro-cckro
    """
    m = _SIM_RE.match(name)
    if not m:
        return {}
    gd = m.groupdict()
    return {
        "dataset": gd["dataset"].lower(),
        "n_items": parse_count_token(gd["n_items"]),
        "n_blocks": int(gd["B"]),
        "top_k": int(gd["top_k"]),
        "p": parse_p_token(gd["p"]),
        "mc": int(gd["mc"]),
        "steps": int(gd["steps"]),
        "d_item": int(gd["d_item"]),
        "d_time": int(gd["d_time"]),
        "suffix": gd.get("suffix") or "",
    }


def dataset_from_name_or_config(out_dir: Path, run_cfg: dict[str, Any], parsed: dict[str, Any]) -> str:
    if parsed.get("dataset"):
        return str(parsed["dataset"]).lower()
    meta = run_cfg.get("prepared_metadata", {}) or {}
    ds = str(meta.get("dataset", "")).lower()
    if ds:
        if "movielens" in ds:
            return "movielens"
        if "retailrocket" in ds:
            return "retailrocket"
        return ds
    name = out_dir.name.lower()
    if "retailrocket" in name:
        return "retailrocket"
    if "movielens" in name:
        return "movielens"
    return name.split("_top", 1)[0]


def sim_folder_from_dir(out_dir: Path) -> SimFolder | None:
    parsed = parse_sim_folder_name(out_dir.name)
    run_cfg = read_json_safely(out_dir / "run_config.json")
    if not parsed and not run_cfg:
        return None
    dataset = dataset_from_name_or_config(out_dir, run_cfg, parsed)
    # Prefer the folder name because old run_config.json can contain stale or partial values.
    def pick(key: str, default: Any) -> Any:
        return parsed.get(key, run_cfg.get(key, default))

    try:
        return SimFolder(
            out_dir=out_dir,
            dataset=dataset,
            n_items=int(pick("n_items", 0)),
            n_blocks=int(pick("n_blocks", parsed.get("B", 0) or run_cfg.get("B", 0))),
            top_k=int(pick("top_k", 0)),
            p=float(pick("p", 0.5)),
            mc=int(pick("mc", 500)),
            steps=int(pick("steps", 400)),
            d_item=int(pick("d_item", 16)),
            d_time=int(pick("d_time", max(int(pick("n_blocks", 1)) - 1, 1))),
            suffix=str(parsed.get("suffix", "")),
            run_config=run_cfg or None,
        )
    except Exception:
        return None


def has_completed_output(path: Path) -> bool:
    return path.is_dir() and ((path / "simulation_summary.csv").exists() or (path / "run_config.json").exists())


def scan_result_root(result_root: Path, *, fallback_to_backup: bool = True) -> tuple[Path, list[SimFolder], list[str]]:
    """Scan a result root for completed simulation folders.

    If ``outputs/sim`` is empty but a sibling ``outputs/sim_backup`` exists, use
    the backup folder and return a warning. This matches the common workflow of
    moving the completed main-grid results into a backup folder before running
    extra baselines.
    """
    warnings: list[str] = []
    roots_to_try = [result_root]
    if fallback_to_backup:
        backup = result_root.parent / f"{result_root.name}_backup"
        if backup != result_root:
            roots_to_try.append(backup)
    for root in roots_to_try:
        if not root.exists():
            continue
        folders = [p for p in sorted(root.iterdir()) if has_completed_output(p)]
        parsed = [sf for sf in (sim_folder_from_dir(p) for p in folders) if sf is not None]
        if parsed:
            if root != result_root:
                warnings.append(f"No completed folders found in {result_root}; using fallback result root {root}.")
            return root, parsed, warnings
    if not result_root.exists():
        warnings.append(f"Result root does not exist: {result_root}")
    else:
        warnings.append(f"No completed simulation folders found in {result_root}.")
    return result_root, [], warnings


def infer_config_for_dataset(dataset: str, run_cfg: dict[str, Any] | None = None) -> str:
    # Keep old config if it is relative and nonempty, but avoid absolute local paths in generated commands.
    cfg = str((run_cfg or {}).get("config") or "").strip()
    if cfg and cfg.lower() != "none" and not Path(cfg).is_absolute():
        return cfg
    ds = dataset.lower()
    if "retailrocket" in ds:
        return "configs/retailrocket_10k.yaml"
    if "movielens" in ds:
        return "configs/movielens10m_10k.yaml"
    return "configs/base.yaml"


def infer_prepared_dir(root: Path, sim: SimFolder, *, prefer_relative: bool = True) -> str:
    """Infer the prepared-data directory for a simulation folder.

    Prefer the current package's standard relative path
    ``data/processed/{dataset}_top{N}_b{B}_k{K}``.  Old run_config.json files
    often contain absolute paths such as ``/Users/...``; those are intentionally
    ignored unless the standard path is unavailable and the absolute path exists.
    """
    standard_name = prepared_dir_name(sim.dataset_key, sim.n_items, sim.n_blocks, sim.top_k)
    rel_candidates = [
        Path("data") / "processed" / standard_name,
        Path("outputs") / "prepared" / standard_name,
    ]
    for rel in rel_candidates:
        if (root / rel).exists():
            return str(rel if prefer_relative else (root / rel).resolve())
    # Still return the standard relative data/processed path, because run_simulation.py
    # will resolve it against the project root and provide a clear error if data are absent.
    if prefer_relative:
        return str(rel_candidates[0])
    return str((root / rel_candidates[0]).resolve())


def existing_designs(summary_path: Path) -> set[str]:
    if pd is None or not summary_path.exists():
        return set()
    try:
        df = pd.read_csv(summary_path, usecols=["design"])
        return set(map(str, df["design"].unique()))
    except Exception:
        return set()


def dataset_matches(sim: SimFolder, requested: str) -> bool:
    if requested == "all":
        return True
    key = "movielens" if requested == "movielens10m" else requested
    return key in sim.dataset_key or key in sim.out_dir.name.lower()


def filter_sim_folders(sims: Iterable[SimFolder], *, dataset: str = "all") -> list[SimFolder]:
    return [s for s in sims if dataset_matches(s, dataset)]
