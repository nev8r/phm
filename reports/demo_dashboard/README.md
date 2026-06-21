# Demo Dashboard

This directory contains a static HTML dashboard for the PHM training demo.

Open `index.html` directly or serve this directory with any static file server. The page embeds the generated JSON data and also writes the same extracts under `data/` for review.

Source scope:

- `reports/feature_analysis/`
- `reports/baseline_results/`
- `reports/non_mlp_baseline_results/`

Excluded by design:

- raw run-output directories
- saved trainer state binaries
- prediction tables
- private source locations
- saved model binaries
