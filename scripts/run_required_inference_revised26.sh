#!/usr/bin/env bash
set -euo pipefail

# Run from the COSTA repository root after applying this package.
PYTHON_BIN="${PYTHON_BIN:-python}"
MC="${MC:-1000}"
MEAN_MC="${MEAN_MC:-500}"
TAIL_MC="${TAIL_MC:-500}"
PAIR_BATCH="${PAIR_BATCH:-10000}"
OUT_ROOT="${OUT_ROOT:-outputs/inference_revised26}"
RR_PREPARED="${RR_PREPARED:-data/processed/retailrocket_top10k_b8_k10}"
ML_PREPARED="${ML_PREPARED:-data/processed/movielens_top10k_b8_k10}"
ARTICLE_ROOT="${ARTICLE_ROOT:-}"

for required in "$RR_PREPARED" "$ML_PREPARED"; do
  if [[ ! -d "$required" ]]; then
    echo "Missing processed-data directory: $required" >&2
    exit 2
  fi
  for file in item_block_weight.npy metadata.json; do
    if [[ ! -f "$required/$file" ]]; then
      echo "Missing $required/$file" >&2
      exit 2
    fi
  done
  if [[ ! -f "$required/competition_A.npz" && ! -f "$required/competition_A.npy" ]]; then
    echo "Missing competition_A.npz or competition_A.npy in $required" >&2
    exit 2
  fi
done

COMMON_ARGS=(
  --models linear
  --p-values 0.5
  --designs independent,rbsd_approx,COSTA,COSTA_IC_lam1
  --mc "$MC"
  --mean-calibration-mc "$MEAN_MC"
  --tail-calibration-mc "$TAIL_MC"
  --pair-batch "$PAIR_BATCH"
  --steps 400
  --d-item 16
  --d-time auto
  --n-far-pairs 50000
  --locality-time-gap 2
  --burn-in 1
  --ci-level 0.95
  --tail-outcome-mode conditional_mean
  --item-shock-sd 0
  --block-shock-sd 0
  --alpha-hat-item-sd 0
  --alpha-hat-block-sd 0
  --alpha-hat-cell-sd 0
  --out-root "$OUT_ROOT"
)

"$PYTHON_BIN" scripts/run_inference_validation_revised26.py \
  --config configs/retailrocket_10k.yaml \
  --prepared-dir "$RR_PREPARED" \
  "${COMMON_ARGS[@]}"

"$PYTHON_BIN" scripts/run_inference_validation_revised26.py \
  --config configs/movielens10m_10k.yaml \
  --prepared-dir "$ML_PREPARED" \
  "${COMMON_ARGS[@]}"

if [[ -n "$ARTICLE_ROOT" ]]; then
  "$PYTHON_BIN" scripts/plot_inference_validation_revised26.py \
    --input-root "$OUT_ROOT" \
    --manuscript-dir "$ARTICLE_ROOT" \
    --p 0.5 \
    --model linear \
    --internal-designs COSTA,COSTA_IC_lam1
else
  echo "ARTICLE_ROOT is empty; CSV outputs were generated but manuscript tables/figures were not generated."
fi
