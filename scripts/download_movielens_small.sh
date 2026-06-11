#!/usr/bin/env bash
set -euo pipefail

# Download the official MovieLens latest-small dataset from GroupLens.
# It is about 1 MB compressed and contains ratings.csv with userId, movieId,
# rating, and timestamp columns.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/data/raw/ml-latest-small}"
mkdir -p "$OUT_DIR"
TMP_ZIP="$OUT_DIR/ml-latest-small.zip"
URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

echo "Downloading MovieLens latest-small to $OUT_DIR"
curl -L "$URL" -o "$TMP_ZIP"
unzip -o "$TMP_ZIP" -d "$OUT_DIR"
# The zip extracts into ml-latest-small/. Move files up one level for convenience.
if [ -d "$OUT_DIR/ml-latest-small" ]; then
  cp -f "$OUT_DIR/ml-latest-small"/* "$OUT_DIR"/
fi

echo "Done. Expected ratings file: $OUT_DIR/ratings.csv"
