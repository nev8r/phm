# Final Feature Decisions

## 1. Main Feature Set

Use:

```text
manual_basic
```

for the current downstream baseline experiments.

Rationale:

- It runs on both XJTU-SY and PHM2012.
- It produces interpretable amplitude, dispersion, peak-to-peak, and spectral summary features.
- It supports all three tasks: RUL, Health State, and Early Fault Detection.
- It avoids the XJTU-SY full-size tsfresh memory issue.
- PHM2012 tsfresh adds mostly redundant RMS/std/variance-style features.

## 2. Deferred Feature Sets

`manual_tsfresh_basic`:

- XJTU-SY: blocked by tsfresh long-format memory pressure.
- PHM2012: runs successfully but mostly duplicates manual amplitude and statistical features.
- Decision: not adopted for the current mainline.

`tsfresh_efficient`:

- Not evaluated in this cycle.
- Defer until there is a memory-aware extraction path or an explicit tsfresh engineering stage.

## 3. Recommended Features by Task

### RUL

Use amplitude and energy-like time-domain features.

Primary families:

- magnitude mean / mean absolute
- magnitude standard deviation
- horizontal mean absolute
- horizontal RMS / standard deviation
- vertical amplitude / dispersion as secondary support

Dataset-specific choices:

- XJTU-SY: prioritize `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std`.
- PHM2012: prioritize `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs`.

### Health State

Use amplitude features, with dataset-aware channel preference.

Dataset-specific choices:

- XJTU-SY: magnitude features are safest under cross-condition checks.
- PHM2012: horizontal amplitude features are strongest.

Recommended features:

- XJTU-SY: `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std`, plus horizontal secondary features.
- PHM2012: `h__time__mean_abs`, `h__time__std`, `h__time__rms`, plus magnitude secondary features.

### Early Fault

Use amplitude features, but treat XJTU-SY with stronger condition caveats.

Dataset-specific choices:

- XJTU-SY: prefer magnitude features as the safest global baseline and keep horizontal/spectral features as condition-sensitive.
- PHM2012: horizontal and magnitude amplitude features are strong and more stable.

Recommended features:

- XJTU-SY: `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std`, `v__time__std`, `v__time__mean_abs`.
- PHM2012: `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs`, `h__time__std`, `h__time__rms`.

## 4. Reference Features

`mag__time__rms` is a reference feature, not an independent candidate.

Reason:

- It is the actual HI/FPT source feature in the completed analyses.
- HealthState and EarlyFault pseudo-labels depend on HI/FPT construction.
- Strong scores for this feature partly validate the label construction rather than proving independent predictive value.

## 5. Caveats

- HealthState and EarlyFault are pseudo-label tasks derived from HI/FPT logic.
- XJTU-SY EarlyFault is strongly condition-sensitive.
- PHM2012 official split has larger val/test variance for several amplitude features.
- No bearing-characteristic-frequency features such as BPFO, BPFI, BSF, or FTF were engineered in this cycle.
- No model training or baseline evaluation is included in this feature-analysis phase.

## 6. Next Step

Proceed to downstream baseline planning with:

```text
feature_set = manual_basic
recommendation_source = reports/feature_analysis/summary/recommended_features.csv
```
