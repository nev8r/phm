# Baseline Preflight

## 1. Purpose

Step P verifies that the baseline planning matrix can be translated into task datasets before running any training.

It uses:

```text
mode=inspect_task
```

It does not train models, evaluate models, export predictions, create checkpoints, or produce baseline metrics.

## 2. Commands

### P1. XJTU RUL `full_manual_basic`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=rul_tabular \
  model=mlp \
  run.name=xjtu_main_rul_full_manual_basic_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/xjtu
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-091805_xjtu_main_rul_full_manual_basic_preflight_c26ec29f`
- Report directory: `reports/baseline_planning/preflight/xjtu_main_rul_full_manual_basic`

### P2. XJTU RUL `full_manual_basic_no_reference`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=rul_tabular \
  model=mlp \
  run.name=xjtu_main_rul_full_manual_basic_no_reference_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/xjtu \
  'task.feature_columns.exclude_columns=[mag__time__rms]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-091936_xjtu_main_rul_full_manual_basic_no_reference_preflight_2e2f2043`
- Report directory: `reports/baseline_planning/preflight/xjtu_main_rul_full_manual_basic_no_reference`

### P3. XJTU RUL `compact_non_label_source`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=rul_tabular \
  model=mlp \
  run.name=xjtu_main_rul_compact_non_label_source_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/xjtu \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,v__time__mean_abs,v__time__std]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092103_xjtu_main_rul_compact_non_label_source_preflight_b84bf8b5`
- Report directory: `reports/baseline_planning/preflight/xjtu_main_rul_compact_non_label_source`

### P4. XJTU RUL `compact_with_reference`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=rul_tabular \
  model=mlp \
  run.name=xjtu_main_rul_compact_with_reference_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/xjtu \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,v__time__mean_abs,v__time__std,mag__time__rms]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092229_xjtu_main_rul_compact_with_reference_preflight_52572d1b`
- Report directory: `reports/baseline_planning/preflight/xjtu_main_rul_compact_with_reference`

### P5. XJTU Health State `compact_non_label_source`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=health_state_tabular \
  model=mlp \
  run.name=xjtu_main_health_compact_non_label_source_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/xjtu \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,h__time__rms]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092356_xjtu_main_health_compact_non_label_source_preflight_fdd86c5c`
- Report directory: `reports/baseline_planning/preflight/xjtu_main_health_compact_non_label_source`

### P6. XJTU Early Fault `compact_non_label_source`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=early_fault_tabular \
  model=mlp \
  run.name=xjtu_main_early_compact_non_label_source_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/xjtu \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,v__time__std,v__time__mean_abs]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092532_xjtu_main_early_compact_non_label_source_preflight_ed52e56c`
- Report directory: `reports/baseline_planning/preflight/xjtu_main_early_compact_non_label_source`

### P7. PHM2012 RUL `compact_non_label_source`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=rul_tabular \
  model=mlp \
  run.name=phm_official_rul_compact_non_label_source_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/phm2012 \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[h__time__mean_abs,mag__time__mean,mag__time__mean_abs,h__time__rms,h__time__std,v__time__mean_abs,mag__time__std]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092716_phm_official_rul_compact_non_label_source_preflight_79baa1e2`
- Report directory: `reports/baseline_planning/preflight/phm_official_rul_compact_non_label_source`

### P8. PHM2012 Health State `compact_non_label_source`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=health_state_tabular \
  model=mlp \
  run.name=phm_official_health_compact_non_label_source_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/phm2012 \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[h__time__mean_abs,h__time__std,h__time__rms,mag__time__mean,mag__time__mean_abs]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092823_phm_official_health_compact_non_label_source_preflight_39bc4b2a`
- Report directory: `reports/baseline_planning/preflight/phm_official_health_compact_non_label_source`

### P9. PHM2012 Early Fault `compact_non_label_source`

```bash
uv run bp --config-name smoke \
  mode=inspect_task \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_basic \
  label=degradation_three_tasks \
  task=early_fault_tabular \
  model=mlp \
  run.name=phm_official_early_compact_non_label_source_preflight \
  project.artifact_root=artifacts/baseline_preflight \
  dataset.root=data/loader_roots/phm2012 \
  task.feature_columns.include=patterns \
  'task.feature_columns.include_patterns=[h__time__mean_abs,mag__time__mean,mag__time__mean_abs,h__time__std,h__time__rms,v__time__mean_abs,v__time__std]'
```

- Run directory: `artifacts/baseline_preflight/runs/20260621-092931_phm_official_early_compact_non_label_source_preflight_6034974f`
- Report directory: `reports/baseline_planning/preflight/phm_official_early_compact_non_label_source`

## 3. Preflight Summary

| ID | Dataset | Split | Task | Feature Subset | Feature Count | Target Columns | Label Source Included | Train / Val / Test | Status |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | --- |
| P1 | xjtu_sy | xjtu_bearing_index_split | RUL | full_manual_basic | 45 | `piecewise_rul_norm` | yes | 7032 / 1679 / 505 | pass |
| P2 | xjtu_sy | xjtu_bearing_index_split | RUL | full_manual_basic_no_reference | 44 | `piecewise_rul_norm` | no | 7032 / 1679 / 505 | pass |
| P3 | xjtu_sy | xjtu_bearing_index_split | RUL | compact_non_label_source | 7 | `piecewise_rul_norm` | no | 7032 / 1679 / 505 | pass |
| P4 | xjtu_sy | xjtu_bearing_index_split | RUL | compact_with_reference | 8 | `piecewise_rul_norm` | yes | 7032 / 1679 / 505 | pass |
| P5 | xjtu_sy | xjtu_bearing_index_split | HealthState | compact_non_label_source | 6 | `health_state_id` | no | 7032 / 1679 / 505 | pass |
| P6 | xjtu_sy | xjtu_bearing_index_split | EarlyFault | compact_non_label_source | 5 | `early_fault` | no | 7032 / 1679 / 505 | pass |
| P7 | phm2012 | phm2012_official | RUL | compact_non_label_source | 7 | `piecewise_rul_norm` | no | 7534 / 4330 / 13025 | pass |
| P8 | phm2012 | phm2012_official | HealthState | compact_non_label_source | 5 | `health_state_id` | no | 7534 / 4330 / 13025 | pass |
| P9 | phm2012 | phm2012_official | EarlyFault | compact_non_label_source | 7 | `early_fault` | no | 7534 / 4330 / 13025 | pass |

## 4. Task Type Checks

| ID | Task | Task Type | Expected | Status |
| --- | --- | --- | --- | --- |
| P1 | RUL | regression | regression | pass |
| P2 | RUL | regression | regression | pass |
| P3 | RUL | regression | regression | pass |
| P4 | RUL | regression | regression | pass |
| P5 | HealthState | multiclass_classification | multiclass_classification | pass |
| P6 | EarlyFault | binary_classification | binary_classification | pass |
| P7 | RUL | regression | regression | pass |
| P8 | HealthState | multiclass_classification | multiclass_classification | pass |
| P9 | EarlyFault | binary_classification | binary_classification | pass |

## 5. Checks

| Check | Result | Notes |
| --- | ---: | --- |
| all inspect_task commands succeeded | pass | 9/9 commands completed. |
| RUL target is `piecewise_rul_norm` | pass | P1, P2, P3, P4, P7. |
| HealthState target is `health_state_id` | pass | P5, P8. |
| EarlyFault target is `early_fault` | pass | P6, P9. |
| `no_reference` excludes `mag__time__rms` | pass | P2 has 44 features and no reference feature. |
| `compact_non_label_source` excludes `mag__time__rms` | pass | P3, P5, P6, P7, P8, P9 exclude it. |
| `compact_with_reference` includes `mag__time__rms` | pass | P4 has 8 features and includes it. |
| train/val/test splits are non-empty | pass | All inspected reports have non-zero train, val, and test counts. |
| no training artifacts generated for reports | pass | Curated report copies contain only command/spec/report/feature columns/target columns. |

## 6. Copied Files

Each preflight report directory contains:

```text
command.txt
task_spec.json
task_report.json
feature_columns.txt
target_columns.txt
```

The following raw task files were intentionally not copied:

```text
task_manifest.parquet
task_manifest.csv
```

The following raw artifact directories were intentionally not copied:

```text
features/
labels/
hi/
checkpoints/
predictions/
```

## 7. Issues / Warnings

- Missing features: none detected.
- Unexpected target: none detected.
- Empty split: none detected.
- Feature count mismatch: none detected.
- Other: raw artifact directories under `artifacts/baseline_preflight/` are generated working outputs and are not committed.

## 8. Decision

- [x] Pass
- [ ] Needs fix
- [ ] Blocked

Next action:

```text
Submit Step P for review. If accepted, proceed to Step Q first baseline training smoke.
```
