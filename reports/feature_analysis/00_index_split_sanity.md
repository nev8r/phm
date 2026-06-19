# Step D: Index and Split Sanity Check

## 1. Purpose

Verify that sample index generation and bearing-level splits are correct before feature extraction and feature analysis.

Step D does not extract features, build labels, run feature analysis, or train models.

## 2. Commands

### D1. XJTU-SY bearing-index split

```bash
uv run bp --config-name smoke \
  mode=build_index \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  run.name=step_d_xjtu_bearing_index_split \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

- Run directory: `artifacts/feature_analysis/runs/20260619-180939_step_d_xjtu_bearing_index_split_0cd7b5a3`
- `index_report.ok`: true
- `split_report.ok`: true

### D2. XJTU-SY Condition 1 LOO

```bash
uv run bp --config-name smoke \
  mode=build_index \
  dataset=xjtu_sy \
  split=xjtu_leave_one_bearing_out \
  run.name=step_d_xjtu_c1_leave_one_bearing_out \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu \
  split.condition_id=35Hz12kN \
  split.test_bearing_id=Bearing1_5 \
  split.val_bearing_id=Bearing1_4
```

- Run directory: `artifacts/feature_analysis/runs/20260619-180939_step_d_xjtu_c1_leave_one_bearing_out_5a66b694`
- `index_report.ok`: true
- `split_report.ok`: true

### D3. XJTU-SY Condition 2 LOO

```bash
uv run bp --config-name smoke \
  mode=build_index \
  dataset=xjtu_sy \
  split=xjtu_leave_one_bearing_out \
  run.name=step_d_xjtu_c2_leave_one_bearing_out \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu \
  split.condition_id=37.5Hz11kN \
  split.test_bearing_id=Bearing2_5 \
  split.val_bearing_id=Bearing2_4
```

- Run directory: `artifacts/feature_analysis/runs/20260619-180939_step_d_xjtu_c2_leave_one_bearing_out_9c422007`
- `index_report.ok`: true
- `split_report.ok`: true

### D4. XJTU-SY Condition 3 LOO

```bash
uv run bp --config-name smoke \
  mode=build_index \
  dataset=xjtu_sy \
  split=xjtu_leave_one_bearing_out \
  run.name=step_d_xjtu_c3_leave_one_bearing_out \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu \
  split.condition_id=40Hz10kN \
  split.test_bearing_id=Bearing3_5 \
  split.val_bearing_id=Bearing3_4
```

- Run directory: `artifacts/feature_analysis/runs/20260619-180939_step_d_xjtu_c3_leave_one_bearing_out_baa7472a`
- `index_report.ok`: true
- `split_report.ok`: true

### D5. XJTU-SY cross-condition

```bash
uv run bp --config-name smoke \
  mode=build_index \
  dataset=xjtu_sy \
  split=xjtu_cross_condition \
  run.name=step_d_xjtu_cross_condition \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

- Run directory: `artifacts/feature_analysis/runs/20260619-180939_step_d_xjtu_cross_condition_0cb8b840`
- `index_report.ok`: true
- `split_report.ok`: true

### D6. PHM2012 official

```bash
uv run bp --config-name smoke \
  mode=build_index \
  dataset=phm2012 \
  split=phm2012_official \
  run.name=step_d_phm2012_official_split \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/phm2012
```

- Run directory: `artifacts/feature_analysis/runs/20260619-180939_step_d_phm2012_official_split_3da740d5`
- `index_report.ok`: true
- `split_report.ok`: true

## 3. Split Summary

| Scenario | Train Bearings | Val Bearings | Test Bearings | Train Samples | Val Samples | Test Samples | Status |
|---|---|---|---|---:|---:|---:|---|
| XJTU bearing-index | Bearing1_1, Bearing1_2, Bearing1_3, Bearing2_1, Bearing2_2, Bearing2_3, Bearing3_1, Bearing3_2, Bearing3_3 | Bearing1_4, Bearing2_4, Bearing3_4 | Bearing1_5, Bearing2_5, Bearing3_5 | 7032 | 1679 | 505 | pass |
| XJTU C1 LOO | Bearing1_1, Bearing1_2, Bearing1_3 | Bearing1_4 | Bearing1_5 | 442 | 122 | 52 | pass |
| XJTU C2 LOO | Bearing2_1, Bearing2_2, Bearing2_3 | Bearing2_4 | Bearing2_5 | 1185 | 42 | 339 | pass |
| XJTU C3 LOO | Bearing3_1, Bearing3_2, Bearing3_3 | Bearing3_4 | Bearing3_5 | 5405 | 1515 | 114 | pass |
| XJTU cross-condition | Bearing1_1, Bearing1_2, Bearing1_3, Bearing1_4, Bearing1_5 | Bearing2_1, Bearing2_2, Bearing2_3, Bearing2_4, Bearing2_5 | Bearing3_1, Bearing3_2, Bearing3_3, Bearing3_4, Bearing3_5 | 616 | 1566 | 7034 | pass |
| PHM2012 official | Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2, Bearing3_1, Bearing3_2 | Bearing1_3, Bearing2_3 | Bearing1_4, Bearing1_5, Bearing1_6, Bearing1_7, Bearing2_4, Bearing2_5, Bearing2_6, Bearing2_7, Bearing3_3 | 7534 | 4330 | 13025 | pass |

## 4. Sanity Checks

| Check | Result | Notes |
|---|---:|---|
| XJTU main split has all three conditions | pass | Train/val/test include Bearing1_*, Bearing2_*, and Bearing3_* groups as configured. |
| XJTU bearing-index train suffixes are 1/2/3 | pass | Train bearings are Bearing*_1, Bearing*_2, and Bearing*_3. |
| XJTU bearing-index val suffix is 4 | pass | Val bearings are Bearing1_4, Bearing2_4, and Bearing3_4. |
| XJTU bearing-index test suffix is 5 | pass | Test bearings are Bearing1_5, Bearing2_5, and Bearing3_5. |
| XJTU condition-wise splits are non-overlapping | pass | All C1/C2/C3 split reports have no sample or bearing overlap. |
| XJTU cross-condition split separates operating conditions | pass | Train uses Bearing1_*, val uses Bearing2_*, test uses Bearing3_*. |
| PHM2012 official split matches explicit config | pass | Train/val/test bearings match phm2012_official.yaml. |
| No train/val/test bearing overlap | pass | All split reports have no_bearing_overlap=true. |
| No train/val/test sample overlap | pass | All split reports have no_sample_overlap=true. |

## 5. Files Copied

Selected reproducible JSON reports were copied to:

```text
reports/feature_analysis/index_split_sanity/
```

For each scenario, the copied files are:

```text
<run_name>_index_report.json
<run_name>_split.json
<run_name>_split_report.json
```

`index/sample_index.parquet` and `index/sample_index.csv` were intentionally not copied.

## 6. Issues / Warnings

- Missing bearings: none observed.
- Empty split: none observed.
- Unexpected sample count: none observed for the current local dataset roots.
- Other: repo-local roots were used, consistent with Step C.

## 7. Decision

- [x] Pass
- [ ] Needs fix
- [ ] Blocked

Next action: Step E, feature extraction sanity with `manual_basic`.
