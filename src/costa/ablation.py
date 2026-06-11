from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from .config import read_yaml_config
from .naming import prepared_dir_name


def fmt_list(xs: Iterable[Any] | None) -> str:
    if xs is None:
        return ""
    return ",".join(str(x) for x in xs)


def parse_base_config_overrides(overrides: list[str] | None) -> list[str] | None:
    """Parse optional --base-config overrides.

    Accept repeated flags and comma-separated values. This lets the same
    ablation grid be run for both datasets together or for one dataset at a time
    without editing configs/ablation_grid.yaml.
    """
    if not overrides:
        return None
    out: list[str] = []
    for value in overrides:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or None


def base_configs(grid: dict[str, Any], overrides: list[str] | None = None) -> list[str]:
    if overrides:
        return [str(x) for x in overrides]
    if "base_configs" in grid and grid["base_configs"]:
        return [str(x) for x in grid["base_configs"]]
    return [str(grid.get("base_config", "configs/retailrocket_10k.yaml"))]


def base_data_args(root: Path, base_config: str | Path) -> dict[str, Any]:
    cfg_path = root / base_config if not Path(base_config).is_absolute() else Path(base_config)
    cfg = read_yaml_config(cfg_path)
    data = dict(cfg.get("data", {}))
    sim = dict(cfg.get("simulation", {}))
    return {
        "config": str(base_config),
        "dataset": data.get("dataset", cfg.get("dataset", "retailrocket")),
        "n_items": int(data.get("n_items", cfg.get("n_items", 10000))),
        "n_blocks": int(data.get("n_blocks", cfg.get("n_blocks", 4))),
        "top_k": int(data.get("top_k", cfg.get("top_k", 10))),
        "embed_dim": int(data.get("embed_dim", cfg.get("embed_dim", 64))),
        "output_root": data.get("output_root", cfg.get("output_root", "data/processed")),
        "include_embed_dim_in_name": bool(data.get("include_embed_dim_in_name", False)),
        "p": float(sim.get("p", cfg.get("p", 0.5))),
    }


def prepared_path(root: Path, base: dict[str, Any], *, n_items: int | None = None, n_blocks: int | None = None, top_k: int | None = None) -> Path:
    N = int(base["n_items"] if n_items is None else n_items)
    B = int(base["n_blocks"] if n_blocks is None else n_blocks)
    K = int(base["top_k"] if top_k is None else top_k)
    embed = int(base["embed_dim"]) if base.get("include_embed_dim_in_name", False) else None
    name = prepared_dir_name(str(base["dataset"]), N, B, K, embed)
    out = Path(str(base["output_root"])) / name
    if not out.is_absolute():
        out = root / out
    return out.resolve()


def grid_d_time(grid: dict[str, Any], B: int) -> int:
    raw = grid.get("costa_d_time", None)
    if raw is None:
        return max(1, int(B) - 1)
    if isinstance(raw, str):
        key = raw.strip().lower().replace("_", "-")
        if key in {"", "none", "null", "auto", "b-1", "bminus1", "b-minus-1", "bminus-1"}:
            return max(1, int(B) - 1)
    return int(raw)


def normalize_only(only: str | None, all_flag: bool = False) -> str:
    if all_flag or only is None:
        return "all"
    return str(only)


def iter_prepare_specs(root: Path, grid: dict[str, Any], *, only: str = "all", base_config_overrides: list[str] | None = None) -> list[dict[str, Any]]:
    """Return deduplicated prepared-data specs required by an ablation grid.

    A spec is identified by (base_config, N, B, K).  Treatment probability and
    Kronecker dimension ablations reuse already prepared data and therefore do
    not introduce additional graph construction beyond the default N/K for each B.
    """
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()

    def add(base_config: str, base: dict[str, Any], *, N: int | None = None, B: int | None = None, K: int | None = None, family: str) -> None:
        N_eff = int(base["n_items"] if N is None else N)
        B_eff = int(base["n_blocks"] if B is None else B)
        K_eff = int(base["top_k"] if K is None else K)
        key = (str(base_config), N_eff, B_eff, K_eff)
        if key in seen:
            return
        seen.add(key)
        specs.append({
            "base_config": str(base_config),
            "dataset": str(base["dataset"]),
            "N": N_eff,
            "B": B_eff,
            "K": K_eff,
            "family": family,
            "prepared_dir": str(prepared_path(root, base, n_items=N_eff, n_blocks=B_eff, top_k=K_eff)),
        })

    for base_config in base_configs(grid, base_config_overrides):
        base = base_data_args(root, base_config)
        B_values = [int(x) for x in grid.get("B_values", [base["n_blocks"]])]
        cross_B = bool(grid.get("cross_B_for_all_ablations", True))
        B_loop = B_values if cross_B else [int(base["n_blocks"])]

        if only in {"item_count", "all"}:
            for B in B_loop:
                for N in grid.get("item_counts", [base["n_items"]]):
                    add(base_config, base, N=int(N), B=int(B), family="item_count")

        if only in {"top_k", "all"}:
            for B in B_loop:
                for K in grid.get("top_k", [base["top_k"]]):
                    add(base_config, base, B=int(B), K=int(K), family="top_k")

        # COSTA dimension and p ablations require only default N/K for each B.
        if only in {"d_item", "all"}:
            for B in B_loop:
                add(base_config, base, B=int(B), family="d_item")

        if only in {"p", "all"}:
            for B in B_loop:
                add(base_config, base, B=int(B), family="p")

        if only in {"B", "all"}:
            for B in B_values:
                add(base_config, base, B=int(B), family="B")

    return specs


def prepare_command(spec: dict[str, Any], python_exe: str | None = None) -> list[str]:
    return [
        python_exe or sys.executable,
        "scripts/prepare_data.py",
        "--config", str(spec["base_config"]),
        "--n-items", str(spec["N"]),
        "--n-blocks", str(spec["B"]),
        "--top-k", str(spec["K"]),
    ]


def iter_run_commands(root: Path, grid: dict[str, Any], *, only: str = "all", python_exe: str | None = None, base_config_overrides: list[str] | None = None) -> list[list[str]]:
    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(cmd: list[str]) -> None:
        key = tuple(cmd)
        if key not in seen:
            commands.append(cmd)
            seen.add(key)

    py = python_exe or sys.executable
    for base_config in base_configs(grid, base_config_overrides):
        base = base_data_args(root, base_config)
        B_values = [int(x) for x in grid.get("B_values", [base["n_blocks"]])]
        cross_B = bool(grid.get("cross_B_for_all_ablations", True))
        B_loop = B_values if cross_B else [int(base["n_blocks"])]

        def prep(*, N: int | None = None, B: int | None = None, K: int | None = None) -> str:
            return str(prepared_path(root, base, n_items=N, n_blocks=B, top_k=K))

        if only in {"item_count", "all"}:
            for B in B_loop:
                for N in grid.get("item_counts", [base["n_items"]]):
                    add([py, "scripts/run_simulation.py", "--config", base_config, "--prepared-dir", prep(N=int(N), B=int(B))])

        if only in {"top_k", "all"}:
            for B in B_loop:
                for K in grid.get("top_k", [base["top_k"]]):
                    add([py, "scripts/run_simulation.py", "--config", base_config, "--prepared-dir", prep(B=int(B), K=int(K))])

        if only in {"d_item", "all"}:
            for B in B_loop:
                prepared = prep(B=int(B))
                d_time = grid_d_time(grid, int(B))
                for d_item in grid.get("costa_d_item", [16, 32, 64]):
                    add([
                        py, "scripts/run_simulation.py", "--config", base_config,
                        "--prepared-dir", prepared,
                        "--designs", "COSTA", "--d-item", str(d_item), "--d-time", str(d_time),
                    ])

        if only in {"p", "all"}:
            default_p = float(base.get("p", 0.5))
            # In the full grid, the default p run is already generated by the
            # default/B commands. Avoid rerunning the same output directory with
            # an explicit --p/--designs command. When --only p is used, keep the
            # default p as the reference level.
            for B in B_loop:
                prepared = prep(B=int(B))
                for p in grid.get("p_values", [0.1, 0.2, 0.5]):
                    p_float = float(p)
                    if only == "all" and abs(p_float - default_p) < 1e-12:
                        continue
                    designs = grid.get("designs_for_p_half" if abs(p_float - 0.5) < 1e-12 else "designs_for_p_nonhalf")
                    add([
                        py, "scripts/run_simulation.py", "--config", base_config,
                        "--prepared-dir", prepared,
                        "--p", str(p), "--designs", fmt_list(designs),
                    ])

        if only in {"B", "all"}:
            for B in B_values:
                add([py, "scripts/run_simulation.py", "--config", base_config, "--prepared-dir", prep(B=int(B))])

    return commands


def prepared_is_ready(path: Path) -> bool:
    """Return True when a prepared-data directory has the files needed by run_simulation.py."""
    path = Path(path)
    return (
        path.exists()
        and path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "interactions.csv").exists()
        and (path / "item_block_weight.npy").exists()
        and ((path / "competition_A.npz").exists() or (path / "competition_A.npy").exists())
    )


def required_prepared_dirs_for_runs(root: Path, grid: dict[str, Any], *, only: str = "all", base_config_overrides: list[str] | None = None) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for spec in iter_prepare_specs(root, grid, only=only, base_config_overrides=base_config_overrides):
        p = Path(str(spec["prepared_dir"])).resolve()
        if p not in seen:
            dirs.append(p)
            seen.add(p)
    return dirs
