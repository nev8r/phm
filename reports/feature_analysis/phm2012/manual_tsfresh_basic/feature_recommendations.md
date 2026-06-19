# Feature Recommendations

## Summary
- Analysis: full_feature_analysis_3tasks
- Feature source: raw
- Ranking scope: train_only
- Ranked features: 44
- Leakage warnings: 1

## RUL
Top features:
1. mag__time__mean
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean.png
2. h__time__rms
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__time__rms.png
3. tsfresh__h__root_mean_square
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Requires train-only scaling and cross-bearing validation before interpretation.
   - plot: figures/curves/tsfresh__h__root_mean_square.png
4. h__time__std
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__time__std.png
5. tsfresh__h__standard_deviation
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Requires train-only scaling and cross-bearing validation before interpretation.
   - plot: figures/curves/tsfresh__h__standard_deviation.png

## Health State
Top features:
1. h__time__std
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__time__std.png
2. tsfresh__h__standard_deviation
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Requires train-only scaling and cross-bearing validation before interpretation.
   - plot: figures/curves/tsfresh__h__standard_deviation.png
3. h__time__rms
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__time__rms.png
4. tsfresh__h__root_mean_square
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Requires train-only scaling and cross-bearing validation before interpretation.
   - plot: figures/curves/tsfresh__h__root_mean_square.png
5. mag__time__mean
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean.png

## Early Fault
Top features:
1. mag__time__mean
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean.png
2. h__time__std
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__time__std.png
3. tsfresh__h__standard_deviation
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Requires train-only scaling and cross-bearing validation before interpretation.
   - plot: figures/curves/tsfresh__h__standard_deviation.png
4. h__time__rms
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__time__rms.png
5. tsfresh__h__root_mean_square
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Requires train-only scaling and cross-bearing validation before interpretation.
   - plot: figures/curves/tsfresh__h__root_mean_square.png

## Fault Type
Top features:
1. h__spectral__centroid
   - reason: highest relative score is for HealthState, but it is outside the top recommendation band
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/h__spectral__centroid.png
2. h__spectral__entropy
   - reason: highest relative score is for HealthState, but it is outside the top recommendation band
   - caveat: Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features.
   - plot: figures/curves/h__spectral__entropy.png
3. h__spectral__rms_frequency
   - reason: highest relative score is for EarlyFault, but it is outside the top recommendation band
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/h__spectral__rms_frequency.png
4. h__time__crest_factor
   - reason: highest relative score is for EarlyFault, but it is outside the top recommendation band
   - caveat: Sensitive to spikes/noise; verify curve stability before using as a RUL main feature.
   - plot: figures/curves/h__time__crest_factor.png
5. h__time__kurtosis
   - reason: highest relative score is for EarlyFault, but it is outside the top recommendation band
   - caveat: Sensitive to spikes/noise; verify curve stability before using as a RUL main feature.
   - plot: figures/curves/h__time__kurtosis.png

## Leakage Warnings
- `mag__time__rms`: Feature was used as HI source for FPT-based labels.
