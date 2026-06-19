# Baseline Output Convention

This file defines where later baseline training outputs should go.

Step O does not create these outputs.

## 1. Raw Run Artifacts

Later baseline runs should write raw artifacts to:

```text
artifacts/baselines/runs/<run_id>/
```

Expected raw files:

```text
command.txt
resolved_config.yaml
metrics.json
predictions_summary.csv
confusion_matrix.png
rul_prediction_plot.png
experiment_report.md
```

The exact files depend on task type.

## 2. Curated Reports

Small curated reports should be copied to:

```text
reports/baseline_results/<experiment_id>/
```

Suggested curated files:

```text
experiment_report.md
metrics_summary.csv
figures/
```

## 3. Git Rules

Do not commit:

```text
checkpoints/
raw_predictions.csv
large prediction parquet/csv files
large tensor dumps
raw training logs
```

Commit only:

- planning docs
- small curated summaries
- small figures needed for review
- final comparison tables

## 4. Required Experiment Report Fields

Every curated report should state:

- `experiment_id`
- dataset
- split
- task
- feature config
- feature subset
- label config
- model config
- trainer config
- command
- output directory
- whether `mag__time__rms` is included
- validation metrics
- test metrics
- caveats

## 5. Naming Convention

Use stable experiment IDs from:

```text
reports/baseline_planning/EXPERIMENT_MATRIX.csv
```

Example:

```text
xjtu_main_rul_mlp_compact_non_label_source
```

Raw run names may include timestamps, but curated report directories should use the stable `experiment_id`.
