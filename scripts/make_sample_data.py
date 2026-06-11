#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main(out: str = "data/sample/interactions.csv", n_users: int = 400, n_items: int = 160, n_events: int = 8000, seed: int = 123):
    rng = np.random.default_rng(seed)
    user_pref = rng.normal(size=(n_users, 8))
    item_emb = rng.normal(size=(n_items, 8))
    logits = user_pref @ item_emb.T / np.sqrt(8)
    item_pop = rng.gamma(1.5, 1.0, size=n_items)
    probs = np.exp(logits / 2.0) * item_pop[None, :]
    probs = probs / probs.sum(axis=1, keepdims=True)
    rows = []
    timestamps = np.sort(rng.integers(1_600_000_000, 1_620_000_000, size=n_events))
    event_types = np.array(["view", "addtocart", "transaction"])
    event_p = np.array([0.92, 0.06, 0.02])
    for t in timestamps:
        u = int(rng.integers(0, n_users))
        i = int(rng.choice(n_items, p=probs[u]))
        e = str(rng.choice(event_types, p=event_p))
        w = {"view": 1.0, "addtocart": 3.0, "transaction": 8.0}[e]
        rows.append((t, f"u{u}", e, f"item{i}", w))
    df = pd.DataFrame(rows, columns=["timestamp", "user_id", "event", "item_id", "event_weight"])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {df.shape}")


if __name__ == "__main__":
    main()
