#!/usr/bin/env bash
set -euo pipefail

# Fetch the original MovieLens 10M release directly from GroupLens.
# The complete upstream package, including its README and license, is retained.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://files.grouplens.org/datasets/movielens/ml-10m.zip"
LICENSE_URL="https://files.grouplens.org/datasets/movielens/ml-10m-README.html"

usage() {
  printf 'Usage: bash %s [output-directory]\n' "${BASH_SOURCE[0]}"
  printf 'Default: %s/data/raw/ml-10m\n' "$ROOT_DIR"
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

OUT_DIR="${1:-$ROOT_DIR/data/raw/ml-10m}"
while [[ "$OUT_DIR" != / && "$OUT_DIR" == */ ]]; do
  OUT_DIR="${OUT_DIR%/}"
done
OUT_NAME="$(basename "$OUT_DIR")"
if [[ "$OUT_NAME" == . || "$OUT_NAME" == .. || "$OUT_DIR" == / ]]; then
  printf 'Choose a dedicated dataset directory, not %s.\n' "$OUT_DIR" >&2
  exit 2
fi

printf 'Source: %s\n' "$URL"
printf 'MovieLens terms: %s\n' "$LICENSE_URL"
printf 'Read the upstream terms before use. Redistribution requires separate permission from GroupLens.\n'

complete_dataset() {
  [[ -s "$1/ratings.dat" && -s "$1/movies.dat" && -s "$1/tags.dat" && -s "$1/README.html" ]]
}

check_destination() {
  if complete_dataset "$OUT_DIR" || complete_dataset "$OUT_DIR/ml-10M100K"; then
    printf 'Existing MovieLens 10M files and upstream README found in %s; skipping download.\n' "$OUT_DIR"
    exit 0
  fi
  if [[ -L "$OUT_DIR" || ( -e "$OUT_DIR" && ! -d "$OUT_DIR" ) ]]; then
    printf 'Destination is not a regular directory: %s. Choose a new output directory.\n' "$OUT_DIR" >&2
    exit 1
  fi
  if [[ -d "$OUT_DIR" ]]; then
    local entries=()
    shopt -s nullglob dotglob
    entries=("$OUT_DIR"/*)
    shopt -u nullglob dotglob
    # Fresh clones contain a tracked, empty .gitkeep directory placeholder.
    if [[ ${#entries[@]} -eq 1 && "${entries[0]}" == "$OUT_DIR/.gitkeep" &&
          -f "${entries[0]}" && ! -L "${entries[0]}" && ! -s "${entries[0]}" ]]; then
      return
    fi
    if [[ ${#entries[@]} -ne 0 ]]; then
      printf 'Refusing to overwrite incomplete or unrecognized data in %s.\n' "$OUT_DIR" >&2
      printf 'Move those files aside or select a new output directory, then rerun.\n' >&2
      exit 1
    fi
  fi
}

check_destination
for tool in curl unzip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'Required command not found: %s. Install it and rerun this script.\n' "$tool" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUT_DIR")"
PARENT_DIR="$(cd "$(dirname "$OUT_DIR")" && pwd)"
OUT_DIR="$PARENT_DIR/$OUT_NAME"
LOCK_DIR="$PARENT_DIR/.$OUT_NAME.download.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'Cannot acquire download lock: %s.\n' "$LOCK_DIR" >&2
  printf 'If no other download is running, remove the stale lock directory and rerun.\n' >&2
  exit 1
fi
STAGING_DIR=""
cleanup() {
  if [[ -n "$STAGING_DIR" ]]; then
    rm -rf "$STAGING_DIR"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
check_destination
STAGING_DIR="$(mktemp -d "$PARENT_DIR/.$OUT_NAME.download.XXXXXX")"

printf 'Downloading MovieLens 10M to %s\n' "$OUT_DIR"
if ! curl --fail --location --show-error --retry 3 --retry-delay 2 \
  --connect-timeout 30 --proto '=https' --proto-redir '=https' \
  --output "$STAGING_DIR/ml-10m.zip" "$URL"; then
  printf 'Download failed. Check network access to GroupLens and rerun; existing data was not modified.\n' >&2
  exit 1
fi
if ! unzip -tq "$STAGING_DIR/ml-10m.zip"; then
  printf 'The downloaded file is not a valid ZIP archive. Rerun to download a fresh copy.\n' >&2
  exit 1
fi
if ! unzip -q "$STAGING_DIR/ml-10m.zip" -d "$STAGING_DIR/unpacked"; then
  printf 'Extraction failed. Check available disk space and rerun.\n' >&2
  exit 1
fi
PACKAGE_DIR="$STAGING_DIR/unpacked/ml-10M100K"
if ! complete_dataset "$PACKAGE_DIR"; then
  printf 'Unexpected GroupLens archive layout: expected ml-10M100K/{ratings.dat,movies.dat,tags.dat,README.html}.\n' >&2
  printf 'Check the official source: %s\n' "$URL" >&2
  exit 1
fi

# Recheck before installation; never merge into or overwrite existing data.
check_destination
if [[ -d "$OUT_DIR" ]]; then
  if [[ -f "$OUT_DIR/.gitkeep" ]]; then
    # Retain the tracked placeholder when replacing its otherwise empty folder.
    mv "$OUT_DIR/.gitkeep" "$PACKAGE_DIR/.gitkeep"
  fi
  rmdir "$OUT_DIR"
fi
mv "$PACKAGE_DIR" "$OUT_DIR"
printf 'Done. Ratings: %s/ratings.dat\n' "$OUT_DIR"
printf 'Upstream README and license: %s/README.html\n' "$OUT_DIR"
