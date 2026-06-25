# CLI Demo Run Outputs

## Command 1

- status: pass
- exit_code: 0

```bash
uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=validate project.artifact_root=/Users/nev8r/Desktop/phm2/reports/cli_demo/artifacts hydra.output_subdir=null
```

### stdout

```text
Validation succeeded. Run directory: /Users/nev8r/Desktop/phm2/reports/cli_demo/artifacts/runs/20260625-192941_smoke_47e94541
```

### stderr

```text
<empty>
```

## Command 2

- status: pass
- exit_code: 0

```bash
uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=build_index dataset=xjtu_sy split=xjtu_leave_one_bearing_out dataset.root=/Users/nev8r/Desktop/phm2/reports/cli_demo/sample_data/xjtu project.artifact_root=/Users/nev8r/Desktop/phm2/reports/cli_demo/artifacts split.condition_id=35Hz12kN split.test_bearing_id=Bearing1_5 split.val_bearing_id=Bearing1_4 hydra.output_subdir=null
```

### stdout

```text
Index build succeeded. Run directory: /Users/nev8r/Desktop/phm2/reports/cli_demo/artifacts/runs/20260625-192942_smoke_a0d70170
```

### stderr

```text
<empty>
```

