#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw/retailrocket
cat <<'MSG'
RetailRocket is hosted on Kaggle and requires a Kaggle account/API token.

1. Install Kaggle CLI: pip install kaggle
2. Place kaggle.json under ~/.kaggle/kaggle.json and chmod 600 it.
3. Then run:

   kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip

After download, data/raw/retailrocket/events.csv should exist.
MSG
