#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costa.result_index import (
    filter_sim_folders,
    infer_config_for_dataset,
    infer_prepared_dir,
    scan_result_root,
)


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Append the two cluster-level COSTA methods to completed simulation folders. "
            "Backward-compatible with old folders and outputs/sim_backup fallback."
        )
    )
    ap.add_argument("--grid", default="configs/ablation_grid.yaml", help="Grid YAML. Used only for default extra design list.")
    ap.add_argument("--result-root", default="outputs/sim", help="Directory containing completed simulation folders.")
    ap.add_argument("--dataset", choices=["all", "retailrocket", "movielens", "movielens10m"], default="all")
    ap.add_argument("--designs", default=None, help="Comma-separated extra designs. Default comes from extra_cluster_costa_designs in the grid.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-progress", action="store_true", help="Forward --no-progress to run_simulation.py.")
    ap.add_argument("--no-plot", action="store_true", help="Forward --no-plot to run_simulation.py.")
    ap.add_argument("--limit", type=int, default=None, help="Run at most this many folders; useful for smoke tests.")
    ap.add_argument("--no-backup-fallback", action="store_true", help="Do not automatically fall back from outputs/sim to outputs/sim_backup.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    grid_path = ROOT / args.grid if not Path(args.grid).is_absolute() else Path(args.grid)
    grid = read_yaml(grid_path)
    extra_designs = args.designs or ",".join(grid.get("extra_cluster_costa_designs", ["COSTA_cluster_no_kro", "COSTA_cluster_kro"]))
    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = ROOT / result_root
    effective_root, sims, warnings = scan_result_root(result_root, fallback_to_backup=not args.no_backup_fallback)
    for msg in warnings:
        print(f"[run_extra_cluster_costa_grid] {msg}")
    sims = filter_sim_folders(sims, dataset=args.dataset)

    commands: list[list[str]] = []
    for sim in sims:
        cfg = infer_config_for_dataset(sim.dataset_key, sim.run_config)
        prepared = infer_prepared_dir(ROOT, sim, prefer_relative=True)
        cmd = [
            sys.executable,
            "scripts/run_simulation.py",
            "--config", cfg,
            "--prepared-dir", prepared,
            "--out-dir", str(sim.out_dir.relative_to(ROOT) if sim.out_dir.is_relative_to(ROOT) else sim.out_dir),
            "--designs", str(extra_designs),
            "--append-results",
            "--p", str(sim.p),
            "--mc", str(sim.mc),
            "--steps", str(sim.steps),
            "--d-item", str(sim.d_item),
            "--d-time", str(sim.d_time),
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        if args.no_plot or bool((sim.run_config or {}).get("no_plot", False)):
            cmd.append("--no-plot")
        commands.append(cmd)

    if args.limit is not None:
        commands = commands[: int(args.limit)]
    print(f"Extra-design append commands: {len(commands)}")
    print(f"Effective result root: {effective_root}")
    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return
    for cmd in commands:
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("Extra cluster-level COSTA append run complete.")


if __name__ == "__main__":
    main()
