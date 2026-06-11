#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costa.config import read_yaml_config
from costa.ablation import (
    iter_run_commands,
    normalize_only,
    parse_base_config_overrides,
    prepared_is_ready,
    required_prepared_dirs_for_runs,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run COSTA ablation simulations using already-prepared data. "
            "Run scripts/prepare_ablation_grid.py first if the required data/processed directories do not exist."
        )
    )
    ap.add_argument("--grid", default="configs/ablation_grid.yaml", help="Ablation grid YAML file.")
    ap.add_argument("--dry-run", action="store_true", help="Print simulation commands without executing them.")
    ap.add_argument("--all", action="store_true", help="Run every ablation family in the grid. Equivalent to --only all.")
    ap.add_argument(
        "--only",
        choices=["item_count", "top_k", "d_item", "p", "B", "all"],
        default="all",
        help="Which ablation family to run.",
    )
    ap.add_argument(
        "--base-config",
        action="append",
        default=None,
        help=(
            "Optional dataset config override. May be repeated or comma-separated. "
            "If omitted, uses base_configs from the grid."
        ),
    )
    ap.add_argument(
        "--allow-missing-prepared", "--skip-missing",
        dest="allow_missing_prepared",
        action="store_true",
        help=(
            "Do not fail early if a required prepared-data directory is missing. "
            "The downstream run_simulation.py command will fail if it cannot load the directory."
        ),
    )
    return ap


def _command_prepared_dir(cmd: list[str]) -> Path | None:
    try:
        idx = cmd.index("--prepared-dir")
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return Path(cmd[idx + 1]).resolve()


def main() -> None:
    args = build_parser().parse_args()
    only = normalize_only(args.only, args.all)
    grid_path = ROOT / args.grid if not Path(args.grid).is_absolute() else Path(args.grid)
    grid = read_yaml_config(grid_path)
    base_overrides = parse_base_config_overrides(args.base_config)

    commands = iter_run_commands(
        ROOT,
        grid,
        only=only,
        python_exe=sys.executable,
        base_config_overrides=base_overrides,
    )

    if not commands:
        print("No simulation commands were generated.")
        return

    required = required_prepared_dirs_for_runs(ROOT, grid, only=only, base_config_overrides=base_overrides)
    missing = [p for p in required if not prepared_is_ready(p)]
    if missing and not args.dry_run and not args.allow_missing_prepared:
        print("ERROR: Missing prepared-data directories.", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        base_arg = ""
        if args.base_config:
            base_arg = " " + " ".join(f"--base-config {x}" for x in args.base_config)
        print("\nRun the matching prepare step first:", file=sys.stderr)
        print(f"  {sys.executable} scripts/prepare_ablation_grid.py --grid {args.grid} --only {only}{base_arg}", file=sys.stderr)
        print("Then rerun this script.", file=sys.stderr)
        raise SystemExit(2)

    if args.dry_run:
        print(f"Simulation commands: {len(commands)}")
        if missing:
            print(f"[dry-run warning] {len(missing)} required prepared-data directorie(s) are currently missing.")
        for cmd in commands:
            p = _command_prepared_dir(cmd)
            status = "prepared" if p is not None and prepared_is_ready(p) else "missing" if p is not None else "unknown"
            print(f"[{status}] {' '.join(cmd)}")
        print("Dry run complete. No simulations were run.")
        return

    print(f"Simulation commands: {len(commands)}")
    for cmd in commands:
        p = _command_prepared_dir(cmd)
        status = "prepared" if p is not None and prepared_is_ready(p) else "missing" if p is not None else "unknown"
        print(f"[{status}] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("Ablation simulation run complete.")


if __name__ == "__main__":
    main()
