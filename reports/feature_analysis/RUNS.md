# Feature Analysis Runs

This file records all feature-analysis runs that are copied into `reports/feature_analysis`.

## Rules

- Every run must have a `run.name`.
- Every run must record the exact command.
- Every run must record the source artifact directory.
- Every run must be listed in `MANIFEST.csv`.

## Planned Runs

| Step | Dataset | Split | Feature | Label | Analysis | Run Name | Status |
|---|---|---|---|---|---|---|---|
| Step C | env | none | none | none | none | environment_check | done |
| Step D | xjtu_sy | xjtu_bearing_index_split | none | none | none | xjtu_index_split_sanity | done |
| Step E | xjtu_sy | xjtu_bearing_index_split | manual_basic | none | none | xjtu_feature_extraction_manual_basic | done |
| Step F | xjtu_sy | xjtu_bearing_index_split | manual_basic | degradation_three_tasks | full_feature_analysis_3tasks | xjtu_all_conditions_3tasks_manual_basic | done |
| Step G | xjtu_sy | xjtu_bearing_index_split | manual_tsfresh_basic | degradation_three_tasks | full_feature_analysis_3tasks | xjtu_all_conditions_3tasks_manual_tsfresh | blocked |
| Step H | xjtu_sy | xjtu_leave_one_bearing_out | manual_basic | degradation_three_tasks | full_feature_analysis_3tasks | xjtu_condition_wise_manual_basic | done |
| Step I | xjtu_sy | xjtu_cross_condition | manual_basic | degradation_three_tasks | full_feature_analysis_3tasks | xjtu_cross_condition_3tasks_manual_basic | needs-review |
| Step J | phm2012 | phm2012_official | manual_basic | degradation_three_tasks | full_feature_analysis_3tasks | phm2012_3tasks_manual_basic | pending |
| Step K | phm2012 | phm2012_official | manual_tsfresh_basic | degradation_three_tasks | full_feature_analysis_3tasks | phm2012_3tasks_manual_tsfresh | pending |

## Status Values

- `pending`: planned but not run yet.
- `done`: run completed and report artifacts were copied.
- `needs-review`: run completed but findings need manual review.
- `blocked`: run could not complete or should not be used.

## Blocked Runs

- Step G is blocked because full XJTU-SY `manual_tsfresh_basic` extraction was killed before feature extraction completed. The current mainline continues with `manual_basic`; full-size tsfresh comparison is deferred.
