# Step Q: First Compact MLP Training Batch

## Scope

Step Q runs the first real baseline-training batch for the two prepared datasets and the three unified tasks. These runs use `mode=train`, not `mode=inspect_task`, not smoke testing, and not dry-run execution.

- Artifact root: `artifacts/baselines`.
- Feature config: `manual_basic`.
- Label config: `degradation_three_tasks`.
- Model: `mlp`.
- Trainer: `base`.
- Epochs: 50.
- Feature subset: compact non-label-source columns selected from Step P.

## Experiments

| ID | Dataset | Split | Task | Features | Best epoch | Test metric |
| --- | --- | --- | --- | ---: | ---: | --- |
| Q1 | XJTU-SY | `xjtu_bearing_index_split` | RUL regression | 7 | 9 | RMSE 0.428591 |
| Q2 | XJTU-SY | `xjtu_bearing_index_split` | Health state classification | 6 | 11 | MacroF1 0.308675 |
| Q3 | XJTU-SY | `xjtu_bearing_index_split` | Early fault classification | 5 | 13 | MacroF1 0.826390 |
| Q4 | PHM2012 | `phm2012_official` | RUL regression | 7 | 50 | RMSE 0.392475 |
| Q5 | PHM2012 | `phm2012_official` | Health state classification | 5 | 1 | MacroF1 0.295758 |
| Q6 | PHM2012 | `phm2012_official` | Early fault classification | 7 | 1 | MacroF1 0.664085 |

## Validation Summary

All six raw run directories contain the required training artifacts:

- `config/resolved.yaml`
- `run.json`
- `validation_report.json`
- `task/task_spec.json`
- `task/task_report.json`
- `task/feature_columns.txt`
- `task/target_columns.txt`
- `metrics/history.json`
- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `trainer/trainer_state.json`
- `trainer/model_summary.txt`
- `report.md`
- `checkpoints/best.ckpt`
- `checkpoints/last.ckpt`
- `predictions/val_predictions.parquet`
- `predictions/test_predictions.parquet`

All six runs have `history.json` length 50, last history epoch 50, and `trainer_state.epoch` 50.

## Curated Outputs

Each experiment directory under `reports/baseline_results/` includes only small review files:

- `command.txt`
- `resolved_config.yaml`
- `task_spec.json`
- `task_report.json`
- `feature_columns.txt`
- `target_columns.txt`
- `history.json`
- `val_metrics.json`
- `test_metrics.json`
- `trainer_state.json`
- `model_summary.txt`
- `experiment_report.md`

Large artifacts remain under `artifacts/baselines/runs/` and are intentionally not copied into the report tree. This includes checkpoints, prediction parquet files, task manifests, raw and cleaned features, labels, HI files, and index files.

## Metric Table

The machine-readable summary is stored in `reports/baseline_results/first_training_batch_metrics.csv`.

## Notes

This batch is a first compact MLP baseline. It proves that the Stage 0 to Stage 5 pipeline can train, checkpoint, evaluate, and export predictions for both datasets and all three tasks. Metric quality should be interpreted as a baseline reference, not as a tuned model result.
