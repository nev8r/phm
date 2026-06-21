# Feature Subset CLI Overrides

This file translates Step O feature-subset definitions into `bp` CLI overrides for Step P and later baseline runs.

## 1. `full_manual_basic`

No feature-column override.

Expected behavior:

- include all cleaned `manual_basic` features
- include `mag__time__rms`
- Step P count: 45

## 2. `full_manual_basic_no_reference`

```bash
'task.feature_columns.exclude_columns=[mag__time__rms]'
```

Expected behavior:

- include all cleaned `manual_basic` features except `mag__time__rms`
- Step P count: 44

## 3. `compact_non_label_source`

Rules:

- include exact feature names as fnmatch patterns
- exclude `mag__time__rms`
- report `label_source_included=no`

### XJTU-SY RUL

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,v__time__mean_abs,v__time__std]'
```

Expected count: 7

### XJTU-SY Health State

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,h__time__rms]'
```

Expected count: 6

### XJTU-SY Early Fault

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,v__time__std,v__time__mean_abs]'
```

Expected count: 5

### PHM2012 RUL

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[h__time__mean_abs,mag__time__mean,mag__time__mean_abs,h__time__rms,h__time__std,v__time__mean_abs,mag__time__std]'
```

Expected count: 7

### PHM2012 Health State

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[h__time__mean_abs,h__time__std,h__time__rms,mag__time__mean,mag__time__mean_abs]'
```

Expected count: 5

### PHM2012 Early Fault

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[h__time__mean_abs,mag__time__mean,mag__time__mean_abs,h__time__std,h__time__rms,v__time__mean_abs,v__time__std]'
```

Expected count: 7

## 4. `compact_with_reference`

Same as the matching `compact_non_label_source` subset, plus:

```text
mag__time__rms
```

For Step P, this is checked on XJTU-SY RUL:

```bash
task.feature_columns.include=patterns \
'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,v__time__mean_abs,v__time__std,mag__time__rms]'
```

Expected behavior:

- include `mag__time__rms`
- report `label_source_included=yes`
- Step P count for XJTU-SY RUL: 8

## 5. Reporting Rules

- `compact_non_label_source` must exclude `mag__time__rms`.
- `compact_with_reference` must include `mag__time__rms`.
- `full_manual_basic` includes `mag__time__rms` by default.
- `full_manual_basic_no_reference` excludes `mag__time__rms`.
- All baseline reports must state whether the reference feature is included.
- Reference-including runs must not be used as independent feature-evidence claims for Health State or Early Fault.
