#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Conservative thread defaults keep the prototype reliable in small containers.
# Override these environment variables externally for large production runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from costa.config import read_yaml_config
from costa.data import (
    load_retailrocket_events,
    load_movielens_ratings,
    load_sample_interactions,
    prepare_interactions,
    build_user_item_matrix,
    save_prepared,
)
from costa.embeddings import cf_item_embedding
from scipy import sparse
from costa.graph import topk_competition_graph
from costa.naming import prepared_dir_name, write_json


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare recommendation data for COSTA simulations.")
    ap.add_argument("--config", default=None, help="Optional YAML config. Only top-level and data-section keys apply here.")
    ap.add_argument("--dataset", choices=["retailrocket", "movielens", "sample"], default="sample")
    ap.add_argument("--raw-dir", default="data/sample")
    ap.add_argument("--sample-file", default="data/sample/interactions.csv")
    ap.add_argument(
        "--out", "--out-dir", dest="out_dir", default=None,
        help="Output directory. If omitted, uses data/processed/{dataset}_top{N}_b{B}_k{K}.",
    )
    ap.add_argument("--output-root", default="data/processed")
    ap.add_argument("--n-items", type=int, default=120)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--include-embed-dim-in-name", action="store_true", help="Append _d{embed_dim} to the automatic output directory stem.")
    ap.add_argument("--save-dense-graph", action="store_true", help="Also save competition_A.npy. The sparse competition_A.npz is always saved and preferred for 10k+ items.")
    ap.add_argument("--min-user-events", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    return ap


def _apply_data_config(ap: argparse.ArgumentParser, config_path: str | None) -> None:
    if not config_path:
        return
    cfg = read_yaml_config(config_path)
    flat = {k: v for k, v in cfg.items() if k not in {"data", "simulation", "optimization", "outcome", "outcomes", "monitoring"}}
    if isinstance(cfg.get("data"), dict):
        flat.update(cfg["data"])
    allowed = {a.dest for a in ap._actions}
    ap.set_defaults(**{str(k).replace("-", "_"): v for k, v in flat.items() if str(k).replace("-", "_") in allowed})


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    ap = build_parser()
    _apply_data_config(ap, known.config)
    args = ap.parse_args()

    if args.dataset == "retailrocket":
        df = load_retailrocket_events(args.raw_dir)
    elif args.dataset == "movielens":
        df = load_movielens_ratings(args.raw_dir)
    else:
        df = load_sample_interactions(args.sample_file)

    auto_name = prepared_dir_name(args.dataset, args.n_items, args.n_blocks, args.top_k, args.embed_dim if args.include_embed_dim_in_name else None)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.output_root) / auto_name
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    prep = prepare_interactions(
        df,
        n_items=args.n_items,
        n_blocks=args.n_blocks,
        min_user_events=args.min_user_events,
        seed=args.seed,
    )
    save_prepared(prep, out_dir)
    X = build_user_item_matrix(prep)
    emb = cf_item_embedding(X, dim=args.embed_dim, random_state=args.seed)
    A = topk_competition_graph(emb, k=args.top_k, return_sparse=True)
    np.save(out_dir / "item_embedding.npy", emb)
    sparse.save_npz(out_dir / "competition_A.npz", A)
    # For small smoke-test data, saving a dense copy is convenient. For 10k+ item
    # runs, prefer the sparse npz to avoid 400MB+ dense graph files.
    if args.save_dense_graph or prep.n_items <= 1000:
        np.save(out_dir / "competition_A.npy", A.toarray().astype(np.float32))

    metadata: dict[str, Any] = {
        "dataset": args.dataset,
        "raw_dir": str(args.raw_dir),
        "sample_file": str(args.sample_file) if args.dataset == "sample" else None,
        "n_items_requested": int(args.n_items),
        "n_items_effective": int(prep.n_items),
        "n_users_effective": int(prep.n_users),
        "n_blocks": int(prep.n_blocks),
        "top_k": int(args.top_k),
        "embed_dim": int(args.embed_dim),
        "include_embed_dim_in_name": bool(args.include_embed_dim_in_name),
        "save_dense_graph": bool(args.save_dense_graph),
        "graph_storage": "sparse_npz",
        "min_user_events": int(args.min_user_events),
        "seed": int(args.seed),
        "auto_name": auto_name,
        "run_stem": auto_name,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "metadata.json", metadata)

    print(
        f"Prepared data: dataset={args.dataset}, requested_N={args.n_items}, "
        f"N_effective={prep.n_items}, users={prep.n_users}, B={prep.n_blocks}, K={args.top_k}, D={args.embed_dim}"
    )
    print(f"Saved to {out_dir}")
    if args.out_dir is None:
        print("Output directory was auto-generated from dataset, n-items, B, and top-k.")


if __name__ == "__main__":
    main()
