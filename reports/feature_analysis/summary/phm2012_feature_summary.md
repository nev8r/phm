# PHM2012 Feature Summary

## 1. Scope

- Dataset: PHM2012 bearing run-to-failure dataset.
- Split: `phm2012_official`.
- Tasks: RUL, Health State, Early Fault Detection.
- Feature sets evaluated: `manual_basic` and `manual_tsfresh_basic`.
- Label config: `degradation_three_tasks`.
- Analysis config: `full_feature_analysis_3tasks`.

No new analysis was run for this summary. It consolidates Step J and Step K.

## 2. manual_basic Findings

### RUL

PHM2012 RUL is dominated by horizontal and magnitude amplitude features.

Recommended candidates:

- A-level: `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs`
- B-level: `h__time__rms`, `h__time__std`, `v__time__mean_abs`, `mag__time__std`
- Reference: `mag__time__rms`

The top RUL features have clear amplitude-growth behavior. Spectral features are not the primary RUL family in Step J.

### Health State

PHM2012 HealthState strongly favors horizontal-channel amplitude features.

Recommended candidates:

- A-level: `h__time__mean_abs`, `h__time__std`, `h__time__rms`
- B-level: `mag__time__mean`, `mag__time__mean_abs`
- C-level: `mag__spectral__entropy`
- Reference: `mag__time__rms`

Health-state boxplots show clear separation for the horizontal amplitude family, with magnitude amplitude as a useful secondary group.

### Early Fault

PHM2012 EarlyFault is also amplitude-driven.

Recommended candidates:

- A-level: `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs`
- B-level: `h__time__std`, `h__time__rms`, `v__time__mean_abs`, `v__time__std`
- C-level: `v__spectral__rms_frequency`, `v__spectral__peak_frequency`
- Reference: `mag__time__rms`

Spectral frequency features appear only as secondary EarlyFault signals.

## 3. manual_tsfresh_basic Comparison

Step K successfully ran full-size `manual_tsfresh_basic` on PHM2012. The ranking contains 20 `tsfresh__` features and 44 ranked features in total.

The useful `tsfresh__` features are mostly:

- `tsfresh__h__standard_deviation`
- `tsfresh__h__root_mean_square`
- `tsfresh__h__variance`
- `tsfresh__h__maximum`
- `tsfresh__v__standard_deviation`
- `tsfresh__v__root_mean_square`

These features enter top-10 lists, but they mainly duplicate manual RMS, standard deviation, variance, and simple amplitude statistics. They confirm the amplitude-driven story rather than adding a new feature family.

Dropped tsfresh features:

- `tsfresh__h__length`
- `tsfresh__v__length`

These are constant because every PHM2012 snapshot has the same length.

## 4. Feature Set Decision

- `manual_basic`: accepted as the current PHM2012 mainline feature set.
- `manual_tsfresh_basic`: successful but not adopted as the mainline because it is mostly redundant with manual amplitude/statistical features.

## 5. Final PHM2012 Recommended Features

| Task | A-level | B-level | C-level | Reference |
|---|---|---|---|---|
| RUL | `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs` | `h__time__rms`, `h__time__std`, `v__time__mean_abs`, `mag__time__std` | none | `mag__time__rms` |
| HealthState | `h__time__mean_abs`, `h__time__std`, `h__time__rms` | `mag__time__mean`, `mag__time__mean_abs` | `mag__spectral__entropy` | `mag__time__rms` |
| EarlyFault | `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs` | `h__time__std`, `h__time__rms`, `v__time__mean_abs`, `v__time__std` | `v__spectral__rms_frequency`, `v__spectral__peak_frequency` | `mag__time__rms` |
