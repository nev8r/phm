# <Step ID>: <Step Name>

## 1. Purpose

Describe why this step is run.

## 2. Command

```bash
<exact command>
```

## 3. Config

| Item | Value |
|---|---|
| dataset | |
| split | |
| feature | |
| label | |
| analysis | |
| run.name | |
| artifact_root | |
| fit_scope | |
| feature_source | |

## 4. Run Directory

```text
artifacts/feature_analysis/runs/<run_id>/
```

## 5. Files Copied to Report Directory

```text
<list copied files>
```

## 6. Sanity Checks

| Check | Result | Notes |
|---|---:|---|
| analysis_report.ok | | |
| fit_scope=train_only | | |
| feature_source=raw | | |
| leakage_report checked | | |
| feature_ranking exists | | |
| feature_cards exists | | |
| feature_recommendations exists | | |
| figures exist | | |

## 7. Key Findings

### RUL

- Top features:
- Evidence:
- Caveats:

### Health State

- Top features:
- Evidence:
- Caveats:

### Early Fault

- Top features:
- Evidence:
- Caveats:

## 8. Figures Reviewed

- `rul_top_features.png`
- `degradation_score_heatmap.png`
- `health_state_boxplots.png`
- `early_fault_effects.png`
- selected curves:

## 9. Warnings

- Leakage warnings:
- Data quality warnings:
- Distribution shift warnings:
- Other issues:

## 10. Decision

- [ ] Pass
- [ ] Needs rerun
- [ ] Blocked

Next action:
