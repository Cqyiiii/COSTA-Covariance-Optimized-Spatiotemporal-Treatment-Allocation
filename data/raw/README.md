# Raw data layout

Place the raw datasets here before running the preparation scripts.

Expected layout:

```text
data/raw/
  retailrocket/
    events.csv              # or events.csv.zip; nested Kaggle folders are also searched
  ml-10m/
    ratings.dat             # or ratings.csv; nested MovieLens/Kaggle folders are also searched
```

The code does not ship raw RetailRocket or MovieLens files. Prepared data under `data/processed/` is generated locally by `scripts/prepare_data.py`.
