#!/usr/bin/env bash
set -euo pipefail

# Download the Kaggle-hosted MovieLens 10M package used in the COSTA experiments.
# Source: https://www.kaggle.com/datasets/amirmotefaker/movielens-10m-dataset-latest-version?resource=download
#
# This script uses the official Kaggle CLI rather than curl, because Kaggle dataset
# downloads require authentication. Set up ~/.kaggle/kaggle.json first; see README.md.

RAW_DIR="${1:-data/raw/ml-10m}"
DATASET="amirmotefaker/movielens-10m-dataset-latest-version"
mkdir -p "${RAW_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  cat <<MSG
Cannot find the Kaggle CLI.

Install and configure it first:
  pip install kaggle
  mkdir -p ~/.kaggle
  # put kaggle.json from your Kaggle account page into ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json

Then run:
  kaggle datasets download -d ${DATASET} -p ${RAW_DIR} --unzip

Dataset page:
  https://www.kaggle.com/datasets/amirmotefaker/movielens-10m-dataset-latest-version?resource=download
MSG
  exit 1
fi

kaggle datasets download -d "${DATASET}" -p "${RAW_DIR}" --unzip

python - <<PY
from pathlib import Path
raw = Path("${RAW_DIR}")
candidates = list(raw.glob("ratings.dat")) + list(raw.glob("ratings.csv")) + list(raw.rglob("ratings.dat")) + list(raw.rglob("ratings.csv"))
# Deduplicate while preserving order.
seen = set()
uniq = []
for p in candidates:
    if p.resolve() not in seen:
        uniq.append(p)
        seen.add(p.resolve())
if not uniq:
    raise SystemExit(f"Downloaded files under {raw}, but could not find ratings.dat or ratings.csv recursively. Please inspect the extracted Kaggle package.")
print("Found MovieLens ratings file:", uniq[0])
print("prepare_data.py can read this directory recursively:", raw)
PY
