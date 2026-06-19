# Step C: Environment Check

## 1. Purpose

Verify that the repository, Python environment, CLI, configs, and dataset roots are ready for feature analysis.

Step C does not run feature extraction, label building, or feature analysis. It only checks the environment and validates that the three-task configs can be composed by the CLI.

## 2. Git State

- Commit: `081aaa5`
- Branch: `one`
- Working tree clean before Step C report edits: yes

```bash
git rev-parse --short HEAD
git branch --show-current
git status --short
```

Observed output:

```text
081aaa5
one
```

## 3. Environment

| Item | Result | Notes |
|---|---:|---|
| `uv sync` | pass | `Resolved 165 packages`; `Audited 146 packages` |
| `uv run pytest -q` | pass | `90 passed, 11 warnings in 31.70s` |
| Python version | `3.11.14` | From `uv run python --version` |
| XJTU_SY_ROOT set | no | Ambient shell variable was not set |
| XJTU_SY_ROOT exists | no | Not applicable because the variable was not set |
| PHM2012_ROOT set | no | Ambient shell variable was not set |
| PHM2012_ROOT exists | no | Not applicable because the variable was not set |
| Repo-local XJTU root exists | yes | `data/loader_roots/xjtu` |
| Repo-local PHM2012 root exists | yes | `data/loader_roots/phm2012` |

The two validate commands below were executed from the repository root using repo-local dataset roots. The displayed commands use relative paths to avoid writing private absolute paths into the report.

## 4. Dataset Directory Sanity

### XJTU-SY

```text
data/loader_roots/xjtu
data/loader_roots/xjtu/40Hz10kN
data/loader_roots/xjtu/40Hz10kN/Bearing3_2
data/loader_roots/xjtu/40Hz10kN/Bearing3_5
data/loader_roots/xjtu/40Hz10kN/Bearing3_4
data/loader_roots/xjtu/40Hz10kN/Bearing3_3
data/loader_roots/xjtu/40Hz10kN/Bearing3_1
data/loader_roots/xjtu/35Hz12kN
data/loader_roots/xjtu/35Hz12kN/Bearing1_1
data/loader_roots/xjtu/35Hz12kN/Bearing1_5
data/loader_roots/xjtu/35Hz12kN/Bearing1_2
data/loader_roots/xjtu/35Hz12kN/Bearing1_3
data/loader_roots/xjtu/35Hz12kN/Bearing1_4
data/loader_roots/xjtu/37.5Hz11kN
data/loader_roots/xjtu/37.5Hz11kN/Bearing2_1
data/loader_roots/xjtu/37.5Hz11kN/Bearing2_5
data/loader_roots/xjtu/37.5Hz11kN/Bearing2_2
data/loader_roots/xjtu/37.5Hz11kN/Bearing2_3
data/loader_roots/xjtu/37.5Hz11kN/Bearing2_4
```

### PHM2012

```text
data/loader_roots/phm2012
data/loader_roots/phm2012/Full_Test_Set
data/loader_roots/phm2012/Full_Test_Set/Bearing2_6
data/loader_roots/phm2012/Full_Test_Set/Bearing2_7
data/loader_roots/phm2012/Full_Test_Set/Bearing1_6
data/loader_roots/phm2012/Full_Test_Set/Bearing3_3
data/loader_roots/phm2012/Full_Test_Set/Bearing1_7
data/loader_roots/phm2012/Full_Test_Set/Bearing2_5
data/loader_roots/phm2012/Full_Test_Set/Bearing2_3
data/loader_roots/phm2012/Full_Test_Set/Bearing2_4
data/loader_roots/phm2012/Full_Test_Set/Bearing1_5
data/loader_roots/phm2012/Full_Test_Set/Bearing1_3
data/loader_roots/phm2012/Full_Test_Set/Bearing1_4
data/loader_roots/phm2012/Learning_set
data/loader_roots/phm2012/Learning_set/Bearing2_1
data/loader_roots/phm2012/Learning_set/Bearing1_1
data/loader_roots/phm2012/Learning_set/Bearing3_2
data/loader_roots/phm2012/Learning_set/Bearing2_2
data/loader_roots/phm2012/Learning_set/Bearing1_2
data/loader_roots/phm2012/Learning_set/Bearing3_1
```

## 5. CLI Validate Commands

### XJTU-SY

```bash
uv run bp --config-name smoke \
  mode=validate \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=step_c_validate_xjtu_3tasks \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

- Run directory: `artifacts/feature_analysis/runs/20260619-175735_step_c_validate_xjtu_3tasks_25814cfa`
- `validation_report.ok`: true
- Run files checked: `config/`, `run.json`, `validation_report.json`

### PHM2012

```bash
uv run bp --config-name smoke \
  mode=validate \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=step_c_validate_phm2012_3tasks \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/phm2012
```

- Run directory: `artifacts/feature_analysis/runs/20260619-175735_step_c_validate_phm2012_3tasks_718132ec`
- `validation_report.ok`: true
- Run files checked: `config/`, `run.json`, `validation_report.json`

## 6. Decision

- [x] Pass
- [ ] Needs fix
- [ ] Blocked

## 7. Notes

- Ambient `XJTU_SY_ROOT` and `PHM2012_ROOT` were not set in this shell.
- Repo-local loader roots existed and were used for the two validate commands.
- Generated files under `artifacts/feature_analysis/` are runtime artifacts and are not committed.
- Pytest warnings are third-party deprecation warnings from matplotlib/pyparsing.
