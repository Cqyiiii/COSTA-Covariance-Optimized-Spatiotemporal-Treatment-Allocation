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
    iter_prepare_specs,
    normalize_only,
    parse_base_config_overrides,
    prepare_command,
    prepared_is_ready,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare all data directories required by a COSTA ablation grid. "
            "This script only runs prepare_data.py commands; simulations are run separately with run_ablation_grid.py."
        )
    )
    ap.add_argument("--grid", default="configs/ablation_grid.yaml", help="Ablation grid YAML file.")
    ap.add_argument(
        "--base-config",
        action="append",
        default=None,
        help=(
            "Optional dataset config override. May be repeated or comma-separated. "
            "If omitted, uses base_configs from the grid."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Print prepare commands without executing them.")
    ap.add_argument("--force", action="store_true", help="Re-run preparation even when the target prepared directory already exists and looks complete.")
    ap.add_argument("--all", action="store_true", help="Prepare every ablation family in the grid. Equivalent to --only all.")
    ap.add_argument(
        "--only",
        choices=["item_count", "top_k", "d_item", "p", "B", "all"],
        default="all",
        help="Which ablation family's prepared datasets to generate.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    only = normalize_only(args.only, args.all)
    grid_path = ROOT / args.grid if not Path(args.grid).is_absolute() else Path(args.grid)
    grid = read_yaml_config(grid_path)
    base_overrides = parse_base_config_overrides(args.base_config)
    specs = iter_prepare_specs(ROOT, grid, only=only, base_config_overrides=base_overrides)

    if not specs:
        print("No prepared-data commands were generated.")
        return

    print(f"Prepared-data targets: {len(specs)} unique directories")
    ran = 0
    skipped = 0
    for spec in specs:
        out_dir = Path(str(spec["prepared_dir"]))
        cmd = prepare_command(spec, python_exe=sys.executable)
        if prepared_is_ready(out_dir) and not args.force:
            skipped += 1
            print(f"[skip-existing] {out_dir}")
            if args.dry_run:
                print(" ".join(cmd))
            continue
        print("[prepare]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
        ran += 1

    if args.dry_run:
        print("Dry run complete. No data was prepared.")
    else:
        print(f"Preparation complete. Ran {ran} command(s), skipped {skipped} existing target(s).")


if __name__ == "__main__":
    main()
