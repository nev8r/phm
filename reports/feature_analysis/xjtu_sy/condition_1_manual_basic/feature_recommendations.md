# Feature Recommendations

## Summary
- Analysis: full_feature_analysis_3tasks
- Feature source: raw
- Ranking scope: train_only
- Ranked features: 45
- Leakage warnings: 1

## RUL
Top features:
1. mag__time__mean
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean.png
2. mag__time__mean_abs
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean_abs.png
3. v__time__mean_abs
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/v__time__mean_abs.png
4. mag__time__rms
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming. Used as HI source for FPT-based labels; do not overclaim independent detection ability.
   - plot: figures/curves/mag__time__rms.png
5. v__time__rms
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/v__time__rms.png

## Health State
Top features:
1. v__spectral__entropy
   - reason: separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features.
   - plot: figures/curves/v__spectral__entropy.png
2. v__time__mean_abs
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/v__time__mean_abs.png
3. mag__time__mean
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean.png
4. mag__time__mean_abs
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/mag__time__mean_abs.png
5. v__time__rms
   - reason: high RUL score from correlation/trend metrics; separates health-state label distributions
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/v__time__rms.png

## Early Fault
Top features:
1. v__spectral__entropy
   - reason: separates health-state label distributions; changes between healthy and post-FPT samples
   - caveat: Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features.
   - plot: figures/curves/v__spectral__entropy.png
2. mag__spectral__entropy
   - reason: changes between healthy and post-FPT samples
   - caveat: Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features.
   - plot: figures/curves/mag__spectral__entropy.png
3. mag__spectral__centroid
   - reason: changes between healthy and post-FPT samples
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/mag__spectral__centroid.png
4. h__spectral__entropy
   - reason: changes between healthy and post-FPT samples
   - caveat: Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features.
   - plot: figures/curves/h__spectral__entropy.png
5. mag__spectral__rms_frequency
   - reason: changes between healthy and post-FPT samples
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/mag__spectral__rms_frequency.png

## Fault Type
Top features:
1. h__spectral__bandwidth
   - reason: highest relative score is for EarlyFault, but it is outside the top recommendation band
   - caveat: Confirm with train-only ranking and visual plots before overclaiming.
   - plot: figures/curves/h__spectral__bandwidth.png
2. h__spectral__centroid
   - reason: highest relative score is for EarlyFault, but it is outside the top recommendation band
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/h__spectral__centroid.png
3. h__spectral__entropy
   - reason: changes between healthy and post-FPT samples
   - caveat: Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features.
   - plot: figures/curves/h__spectral__entropy.png
4. h__spectral__peak_frequency
   - reason: highest relative score is for HealthState, but it is outside the top recommendation band
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/h__spectral__peak_frequency.png
5. h__spectral__rms_frequency
   - reason: highest relative score is for EarlyFault, but it is outside the top recommendation band
   - caveat: Can be affected by operating-condition changes.
   - plot: figures/curves/h__spectral__rms_frequency.png

## Leakage Warnings
- `mag__time__rms`: Feature was used as HI source for FPT-based labels.
