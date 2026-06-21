# Baseline Planning Runs

This file records planning-stage deliverables for downstream baseline experiments.

## Runs

| Step | Scope | Output | Status |
| --- | --- | --- | --- |
| Step O | planning | baseline planning docs and experiment matrix | done |
| Step P | preflight | inspect_task checks for baseline tasks and feature subsets | needs-review |

## Status Values

- `needs-review`: deliverable is ready for quality review.
- `done`: deliverable has passed review.
- `blocked`: deliverable cannot proceed without new information or implementation work.

## Notes

Step O is documentation and CSV planning only. It does not run training, evaluation, feature extraction, prediction export, or checkpoint creation.

Step P uses `mode=inspect_task` only. It copies small task specs/reports/column lists into `reports/baseline_planning/preflight/` and does not commit raw artifacts, manifests, checkpoints, predictions, or metrics.
