from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 8.0,
    "rating": 1.0,
    "implicit": 1.0,
}


@dataclass
class PreparedData:
    interactions: pd.DataFrame
    item_map: Dict[object, int]
    user_map: Dict[object, int]
    item_block_weight: np.ndarray  # shape (N, B)
    item_popularity: np.ndarray  # shape (N,)
    n_items: int
    n_users: int
    n_blocks: int


def _map_ids(values: Iterable) -> Tuple[Dict[object, int], np.ndarray]:
    uniques = pd.Index(pd.unique(values))
    mapping = {v: i for i, v in enumerate(uniques)}
    codes = pd.Series(values).map(mapping).to_numpy(dtype=np.int64)
    return mapping, codes


def load_retailrocket_events(raw_dir: str | Path) -> pd.DataFrame:
    """Load RetailRocket events.csv.

    Expected columns: timestamp, visitorid, event, itemid, transactionid.
    The Kaggle dataset must be downloaded manually because it requires a Kaggle account.
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / "events.csv"
    zip_path = raw_dir / "events.csv.zip"
    if path.exists():
        df = pd.read_csv(path)
    elif zip_path.exists():
        # pandas can read a single CSV inside a zip archive directly.
        df = pd.read_csv(zip_path)
    else:
        # Some Kaggle downloads unzip into a nested directory, so search recursively.
        candidates = list(raw_dir.glob("**/events.csv")) + list(raw_dir.glob("**/events.csv.zip"))
        if not candidates:
            raise FileNotFoundError(
                f"Cannot find events.csv or events.csv.zip under {raw_dir}. "
                "Download the RetailRocket Kaggle dataset and place it there."
            )
        df = pd.read_csv(candidates[0])
    keep = ["timestamp", "visitorid", "event", "itemid"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"RetailRocket events.csv missing columns: {missing}")
    df = df[keep].rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    df["event_weight"] = df["event"].map(EVENT_WEIGHTS).fillna(1.0).astype(float)
    return df


def _find_movielens_ratings_file(raw_dir: Path) -> Path:
    """Find a MovieLens ratings file in either official or Kaggle layouts.

    The Kaggle package used in the experiments may unzip files directly under
    raw_dir or under a nested folder such as ml-10M100K/.  We therefore search
    recursively, preferring ratings.dat because MovieLens 10M commonly uses the
    double-colon DAT format.
    """
    ordered_patterns = ["ratings.dat", "**/ratings.dat", "ratings.csv", "**/ratings.csv"]
    seen: set[Path] = set()
    for pattern in ordered_patterns:
        for candidate in raw_dir.glob(pattern):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Cannot find ratings.dat or ratings.csv under {raw_dir}. "
        "For the Kaggle MovieLens 10M package, run scripts/download_movielens10m.sh "
        "or unzip the package under data/raw/ml-10m."
    )


def load_movielens_ratings(raw_dir: str | Path) -> pd.DataFrame:
    """Load MovieLens ratings from official or Kaggle package layouts.

    Supported formats:
      - ratings.dat with columns userId::movieId::rating::timestamp
      - ratings.csv with userId,movieId,rating,timestamp

    The returned dataframe uses an implicit interaction weight equal to rating/5.
    """
    raw_dir = Path(raw_dir)
    ratings_path = _find_movielens_ratings_file(raw_dir)
    if ratings_path.suffix.lower() == ".dat":
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )
    else:
        df = pd.read_csv(ratings_path)
        rename = {}
        if "userId" in df.columns:
            rename["userId"] = "user_id"
        if "movieId" in df.columns:
            rename["movieId"] = "item_id"
        df = df.rename(columns=rename)
        required = {"user_id", "item_id", "rating", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"MovieLens ratings file {ratings_path} missing columns: {sorted(missing)}")
    df["event"] = "rating"
    df["event_weight"] = (df["rating"].astype(float) / 5.0).clip(lower=0.0)
    return df[["timestamp", "user_id", "event", "item_id", "event_weight"]]


def load_sample_interactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    required = {"timestamp", "user_id", "event", "item_id", "event_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample interaction file missing columns: {sorted(missing)}")
    return df


def prepare_interactions(
    df: pd.DataFrame,
    n_items: int = 10000,
    n_blocks: int = 4,
    min_user_events: int = 3,
    seed: int = 123,
) -> PreparedData:
    """Filter, map, and block user-item interactions.

    We select the top-n_items by weighted interaction volume, map user/item IDs to
    consecutive integers, and assign interactions to B time blocks using timestamp quantiles.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    if "event_weight" not in df.columns:
        df["event_weight"] = df["event"].map(EVENT_WEIGHTS).fillna(1.0).astype(float)

    # Select items by weighted popularity.
    pop = df.groupby("item_id")["event_weight"].sum().sort_values(ascending=False)
    selected_items = pop.head(n_items).index
    df = df[df["item_id"].isin(selected_items)].copy()

    # Filter users with enough remaining interactions.
    user_counts = df.groupby("user_id").size()
    good_users = user_counts[user_counts >= min_user_events].index
    df = df[df["user_id"].isin(good_users)].copy()

    # If filtering removed too much, keep at least something deterministic.
    if df.empty:
        raise ValueError("No interactions remain after filtering. Lower n_items or min_user_events.")

    # Map IDs.
    item_map, item_codes = _map_ids(df["item_id"])
    user_map, user_codes = _map_ids(df["user_id"])
    df["item_idx"] = item_codes
    df["user_idx"] = user_codes

    n_items_eff = len(item_map)
    n_users_eff = len(user_map)

    # Split timestamp into B approximately equal-sized blocks.
    # pd.qcut can fail with repeated timestamps, so we rank first.
    ranks = df["timestamp"].rank(method="first").to_numpy()
    df["block"] = pd.qcut(ranks, q=n_blocks, labels=False, duplicates="drop").astype(int)
    # If qcut drops bins, rescale manually.
    if df["block"].max() + 1 < n_blocks:
        order = np.argsort(ranks)
        blocks = np.zeros(len(df), dtype=int)
        chunks = np.array_split(order, n_blocks)
        for b, idx in enumerate(chunks):
            blocks[idx] = b
        df["block"] = blocks

    # Item-block exposure/traffic weights.
    ib = (
        df.groupby(["item_idx", "block"])["event_weight"]
        .sum()
        .reset_index()
    )
    item_block_weight = np.zeros((n_items_eff, n_blocks), dtype=float)
    item_block_weight[ib["item_idx"].to_numpy(), ib["block"].to_numpy()] = ib["event_weight"].to_numpy()
    item_popularity = item_block_weight.sum(axis=1)
    item_popularity = item_popularity + 1e-8 * rng.random(n_items_eff)

    return PreparedData(
        interactions=df,
        item_map=item_map,
        user_map=user_map,
        item_block_weight=item_block_weight,
        item_popularity=item_popularity,
        n_items=n_items_eff,
        n_users=n_users_eff,
        n_blocks=n_blocks,
    )


def build_user_item_matrix(prepared: PreparedData) -> sparse.csr_matrix:
    df = prepared.interactions
    mat = sparse.coo_matrix(
        (
            df["event_weight"].to_numpy(dtype=float),
            (df["user_idx"].to_numpy(dtype=np.int64), df["item_idx"].to_numpy(dtype=np.int64)),
        ),
        shape=(prepared.n_users, prepared.n_items),
    ).tocsr()
    return mat


def save_prepared(prepared: PreparedData, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared.interactions.to_csv(out_dir / "interactions.csv", index=False)
    np.save(out_dir / "item_block_weight.npy", prepared.item_block_weight)
    np.save(out_dir / "item_popularity.npy", prepared.item_popularity)
    pd.Series(prepared.item_map).rename("item_idx").to_csv(out_dir / "item_map.csv")
    pd.Series(prepared.user_map).rename("user_idx").to_csv(out_dir / "user_map.csv")
