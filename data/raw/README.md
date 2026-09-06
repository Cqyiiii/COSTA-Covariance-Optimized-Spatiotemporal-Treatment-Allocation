# Raw data layout

Keep raw datasets here only in your local working checkout. Dataset files and downloaded archives must not be committed or included in repository uploads.

MovieLens 10M requires separate permission for redistribution; see the [official usage license](https://files.grouplens.org/datasets/movielens/ml-10m-README.html). Download from the original provider, with no Kaggle credentials needed:

```bash
bash scripts/download_movielens10m.sh
```

The script preserves the upstream README and license alongside the extracted data. Obtain RetailRocket separately using `bash scripts/download_retailrocket.sh` and the instructions in the root README.

Expected layout:

```text
data/raw/
  retailrocket/
    events.csv              # or events.csv.zip; nested Kaggle folders are also searched
  ml-10m/
    ratings.dat             # or ratings.csv; nested MovieLens/Kaggle folders are also searched
```

The release archives contain no `data/` directory. Prepared data under `data/processed/` and toy data under `data/sample/` are generated locally. Neither is shipped. Run `python scripts/build_release.py` from the repository root to create the data-free public and anonymous distributions; `.gitignore` alone does not filter manual ZIP uploads or remove files already committed in Git history.
