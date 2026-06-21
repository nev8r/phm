# Baseline Planning Runs

This file records planning-stage deliverables for downstream baseline experiments.

## Runs

| Step | Scope | Output | Status |
| --- | --- | --- | --- |
| Step O | planning | baseline planning docs and experiment matrix | done |
| Step P | preflight | inspect_task checks for baseline tasks and feature subsets | done |
| Step Q | training | first compact non-label-source MLP baseline batch | done |
| Step R | training | compact-with-reference ablation MLP baseline batch | done |
| Step S | training | full manual_basic MLP baseline batch | done |
| Step T | summary | baseline Q/R/S summary and final main-split decisions | done |
| Step U | training | XJTU cross-condition recommended-subset robustness batch | done |
| Step V | summary | final baseline report and decisions | needs-review |

## Status Values

- `needs-review`: deliverable is ready for quality review.
- `done`: deliverable has passed review.
- `blocked`: deliverable cannot proceed without new information or implementation work.

## Notes

Step O is documentation and CSV planning only. It does not run training, evaluation, feature extraction, prediction export, or checkpoint creation.

Step P uses `mode=inspect_task` only. It copies small task specs/reports/column lists into `reports/baseline_planning/preflight/` and does not commit raw artifacts, manifests, checkpoints, predictions, or metrics.

Step Q runs six real `mode=train` experiments under `artifacts/baselines` with `trainer=base`, `model=mlp`, `feature=manual_basic`, and compact non-label-source feature subsets. It copies only small review artifacts into `reports/baseline_results/`; checkpoints, predictions, task manifests, feature tables, labels, HI files, and index files remain under the raw artifact root.

Step R runs six real `mode=train` experiments under `artifacts/baselines` with the compact-with-reference subset. It compares Step R against Step Q and records the reference-feature effect of adding `mag__time__rms`; raw checkpoints and predictions remain outside the committed report tree.

Step S runs twelve real `mode=train` experiments under `artifacts/baselines` for `full_manual_basic_no_reference` and `full_manual_basic`. It compares full feature sets against the Step Q/R compact subsets; raw checkpoints and predictions remain outside the committed report tree.

Step T summarizes the curated Step Q/R/S result tables into final main-split / official-split baseline decisions. It does not run training, evaluation, feature extraction, prediction export, or checkpoint creation.

Step U runs three real `mode=train` XJTU-SY cross-condition experiments for the Step T independent recommended feature subsets. It keeps raw checkpoints, predictions, feature tables, labels, HI files, and index files under `artifacts/baselines` and copies only small review artifacts into `reports/baseline_results/`.

Step V summarizes the completed 27-run MLP baseline cycle into final dataset/task decisions. It does not run training, evaluation, feature extraction, prediction export, or checkpoint creation.
