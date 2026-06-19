# Feature Analysis

This directory stores curated feature-analysis reports for the bearing PHM project.

The raw runtime outputs are generated under:

```text
artifacts/feature_analysis/runs/<run_id>/
```

Only selected analysis artifacts are copied into this directory for version control.

## Scope

We analyze three tasks consistently across XJTU-SY and PHM2012:

1. RUL
2. Health State
3. Early Fault Detection

Fault-type analysis is intentionally excluded from the main cross-dataset analysis. XJTU-SY fault-type analysis may be revisited later after adding bearing-physics features such as BPFO, BPFI, BSF, and FTF.

## Core Rules

1. Feature ranking and feature recommendations are computed only on the training split.
2. Validation and test splits are used only for distribution inspection and post-hoc visualization.
3. The default analysis uses raw features for interpretability.
4. Features used as HI/FPT sources must be marked as label-source features.
5. Large generated artifacts such as raw features, cleaned features, checkpoints, and predictions should not be committed.

## Main Configs

```text
label=degradation_three_tasks
analysis=full_feature_analysis_3tasks
```

## Main XJTU-SY Split

```text
split=xjtu_bearing_index_split
train: bearing suffix indices 1, 2, 3
val:   bearing suffix index 4
test:  bearing suffix index 5
```

## Main PHM2012 Split

```text
split=phm2012_official
```

## Report Layout

```text
reports/feature_analysis/
├── xjtu_sy/
├── phm2012/
└── summary/
```

Each analysis run should include:

```text
command.txt
analysis_report.json
leakage_report.json
feature_ranking.csv
feature_cards.csv
feature_recommendations.md
selected figures
```
