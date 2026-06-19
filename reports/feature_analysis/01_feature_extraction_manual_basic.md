# Step E: Feature Extraction Sanity with manual_basic

## 1. Purpose

Verify that `manual_basic` features can be extracted for the two main analysis scenarios:

1. XJTU-SY with `xjtu_bearing_index_split`
2. PHM2012 with `phm2012_official`

Step E does not build labels, run feature analysis, or train models.

## 2. Commands

### E1. XJTU-SY bearing-index manual_basic

```bash
uv run bp --config-name smoke \
  mode=extract_features \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  run.name=step_e_xjtu_bearing_index_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

- Run directory: `artifacts/feature_analysis/runs/20260619-213537_step_e_xjtu_bearing_index_manual_basic_43380b33`
- feature_report copied: `reports/feature_analysis/feature_extraction_sanity/step_e_xjtu_bearing_index_manual_basic_feature_report.json`
- feature_spec copied: `reports/feature_analysis/feature_extraction_sanity/step_e_xjtu_bearing_index_manual_basic_feature_spec.json`
- feature_columns copied: `reports/feature_analysis/feature_extraction_sanity/step_e_xjtu_bearing_index_manual_basic_feature_columns.txt`

### E2. PHM2012 official manual_basic

```bash
uv run bp --config-name smoke \
  mode=extract_features \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_basic \
  run.name=step_e_phm2012_official_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/phm2012
```

- Run directory: `artifacts/feature_analysis/runs/20260619-213712_step_e_phm2012_official_manual_basic_5ab0834c`
- feature_report copied: `reports/feature_analysis/feature_extraction_sanity/step_e_phm2012_official_manual_basic_feature_report.json`
- feature_spec copied: `reports/feature_analysis/feature_extraction_sanity/step_e_phm2012_official_manual_basic_feature_spec.json`
- feature_columns copied: `reports/feature_analysis/feature_extraction_sanity/step_e_phm2012_official_manual_basic_feature_columns.txt`

## 3. Feature Extraction Summary

| Scenario | Samples Expected | Raw Feature Rows | Cleaned Feature Rows | Raw Features | Cleaned Features | Dropped Features | NaN Before | Inf Before | Cleaner Scope | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| XJTU bearing-index | 9216 | 9216 | 9216 | 45 | 45 | 0 | 0 | 0 | train_only | pass |
| PHM2012 official | 24889 | 24889 | 24889 | 45 | 45 | 0 | 0 | 0 | train_only | pass |

Both runs produced `feature_report.ok=true`.

## 4. Feature Spec Check

| Check | XJTU | PHM2012 | Notes |
|---|---:|---:|---|
| `name=manual_basic` | pass | pass | Same feature spec used by both runs. |
| `version=v1` | pass | pass | Spec hash: `8854bd21772f`. |
| backend type `manual_processor` | pass | pass | One backend named `manual_basic`. |
| `include_magnitude=true` | pass | pass | Generates `h`, `v`, and `mag` channels. |
| time features include `rms` | pass | pass | HI source candidates are present. |
| spectral features include `centroid` | pass | pass | RUL/health candidate. |
| spectral features include `rms_frequency` | pass | pass | RUL/health candidate. |
| spectral features include `entropy` | pass | pass | Health/early candidate. |
| cleaner `fit_scope=train_only` | pass | pass | Cleaner parameters are estimated from train split only. |

## 5. Required Columns Check

| Column | XJTU | PHM2012 | Notes |
|---|---:|---:|---|
| `h__time__rms` | pass | pass | HI source candidate |
| `v__time__rms` | pass | pass | HI source candidate |
| `mag__time__rms` | pass | pass | HI source candidate |
| `h__time__std` | pass | pass | degradation candidate |
| `v__time__std` | pass | pass | degradation candidate |
| `mag__time__std` | pass | pass | degradation candidate |
| `h__time__mean_abs` | pass | pass | degradation candidate |
| `v__time__mean_abs` | pass | pass | degradation candidate |
| `mag__time__mean_abs` | pass | pass | degradation candidate |
| `h__time__ptp` | pass | pass | degradation candidate |
| `v__time__ptp` | pass | pass | degradation candidate |
| `mag__time__ptp` | pass | pass | degradation candidate |
| `h__time__kurtosis` | pass | pass | early fault candidate |
| `v__time__kurtosis` | pass | pass | early fault candidate |
| `mag__time__kurtosis` | pass | pass | early fault candidate |
| `h__time__crest_factor` | pass | pass | early fault candidate |
| `v__time__crest_factor` | pass | pass | early fault candidate |
| `mag__time__crest_factor` | pass | pass | early fault candidate |
| `h__time__impulse_factor` | pass | pass | early fault candidate |
| `v__time__impulse_factor` | pass | pass | early fault candidate |
| `mag__time__impulse_factor` | pass | pass | early fault candidate |
| `h__time__clearance_factor` | pass | pass | early fault candidate |
| `v__time__clearance_factor` | pass | pass | early fault candidate |
| `mag__time__clearance_factor` | pass | pass | early fault candidate |
| `h__spectral__centroid` | pass | pass | RUL/health candidate |
| `v__spectral__centroid` | pass | pass | RUL/health candidate |
| `mag__spectral__centroid` | pass | pass | RUL/health candidate |
| `h__spectral__rms_frequency` | pass | pass | RUL/health candidate |
| `v__spectral__rms_frequency` | pass | pass | RUL/health candidate |
| `mag__spectral__rms_frequency` | pass | pass | RUL/health candidate |
| `h__spectral__entropy` | pass | pass | health/early candidate |
| `v__spectral__entropy` | pass | pass | health/early candidate |
| `mag__spectral__entropy` | pass | pass | health/early candidate |

## 6. Dropped Features

### XJTU-SY

```text
none
```

### PHM2012

```text
none
```

No HI source columns were dropped.

## 7. Files Copied

Selected feature-extraction sanity files were copied to:

```text
reports/feature_analysis/feature_extraction_sanity/
```

Copied files:

```text
step_e_xjtu_bearing_index_manual_basic_feature_report.json
step_e_xjtu_bearing_index_manual_basic_feature_spec.json
step_e_xjtu_bearing_index_manual_basic_feature_columns.txt
step_e_phm2012_official_manual_basic_feature_report.json
step_e_phm2012_official_manual_basic_feature_spec.json
step_e_phm2012_official_manual_basic_feature_columns.txt
```

The following generated artifacts were intentionally not copied into reports:

```text
raw_features.parquet
cleaned_features.parquet
raw_features.csv
cleaned_features.csv
cleaner_state.pkl
```

## 8. Issues / Warnings

- Missing required columns: none observed.
- Unexpected NaN/Inf: none observed.
- Feature rows mismatch: none observed.
- Dropped HI source columns: none observed.
- Other: repo-local roots were used, consistent with Step C and Step D.

## 9. Decision

- [x] Pass
- [ ] Needs fix
- [ ] Blocked

Next action: Step F, XJTU-SY main feature analysis with `manual_basic`.
