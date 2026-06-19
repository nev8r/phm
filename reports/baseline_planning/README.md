# Baseline Planning

This directory defines the downstream baseline experiment plan derived from the completed feature-analysis cycle.

No model training results are stored here yet.

Inputs:

- `reports/feature_analysis/summary/recommended_features.csv`
- `reports/feature_analysis/summary/final_feature_decisions.md`
- `reports/feature_analysis/FEATURE_ANALYSIS_REPORT.md`
- `reports/feature_analysis/latex/`

Main feature set:

- `manual_basic`

Tasks:

- RUL regression
- Health State classification
- Early Fault detection

Step O is planning only. It does not run training, evaluation, feature extraction, or prediction export.

## Files

- `BASELINE_PLAN.md`: main planning document.
- `EXPERIMENT_MATRIX.csv`: planned experiment matrix for the first baseline stage.
- `FEATURE_SETS.md`: feature subset definitions and label-source rules.
- `METRICS.md`: task-level metric definitions and reporting rules.
- `OUTPUT_CONVENTION.md`: output and Git handoff conventions for later training stages.
- `RUNS.md`: planning-stage status log.
- `MANIFEST.csv`: planning-stage manifest.
- `templates/`: report templates for later baseline result curation.
