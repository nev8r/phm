# Cross-Dataset Feature Comparison

## 1. Common Patterns

Across XJTU-SY and PHM2012, the most consistent feature family is amplitude or energy-like time-domain features.

RUL:

- Both datasets favor magnitude, horizontal, or vertical amplitude statistics.
- `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std`, `h__time__mean_abs`, `h__time__rms`, and `h__time__std` are the most useful families.

Health State:

- Both datasets favor amplitude features.
- PHM2012 is especially horizontal-channel dominant.
- XJTU-SY main split also favors horizontal amplitude, but cross-condition analysis makes magnitude features safer.

Early Fault:

- Both datasets show amplitude features are useful.
- XJTU-SY is more condition-sensitive, especially for EarlyFault.
- PHM2012 is more consistently amplitude-dominant, with spectral frequency features only secondary.

## 2. Differences

| Aspect | XJTU-SY | PHM2012 | Interpretation |
|---|---|---|---|
| RUL | Stable amplitude features across main, condition-wise, and cross-condition checks | Horizontal and magnitude amplitude dominate | RUL can use `manual_basic` amplitude features on both datasets |
| HealthState | Horizontal amplitude strong in main split but magnitude features survive cross-condition better | Horizontal amplitude features dominate | Use magnitude features for robust XJTU HealthState and horizontal features for PHM2012 |
| EarlyFault | Most condition-sensitive task; C1 spectral entropy, C2 peak-to-peak, C3 horizontal amplitude | Amplitude-driven with secondary spectral frequency features | Treat XJTU EarlyFault with stronger condition caveats |
| tsfresh | Full-size XJTU `manual_tsfresh_basic` blocked by memory pressure | PHM2012 `manual_tsfresh_basic` succeeds but is redundant | Do not adopt tsfresh as current mainline |
| Distribution shift | Cross-condition split shows operating-condition mean shifts | PHM2012 top horizontal means are closer but val/test variance is larger | Keep train-only scaling and split-aware validation |

## 3. Label-Source Caveat

`mag__time__rms` is the actual HI/FPT source feature in the main runs.

This means:

- It can be retained as a reference feature.
- It should not be used as independent evidence for HealthState or EarlyFault claims.
- Reports and downstream baselines should separate it from independent candidate features.

## 4. tsfresh Comparison

XJTU-SY:

- Full-size `manual_tsfresh_basic` is blocked.
- The current backend would construct about 604M long-format rows.
- The mainline remains `manual_basic`.

PHM2012:

- Full-size `manual_tsfresh_basic` succeeds.
- `tsfresh__` features enter top-10 lists.
- The useful `tsfresh__` features are mostly RMS, standard deviation, variance, max, and absolute max variants.
- These features repeat the manual amplitude/statistical story.

Final decision:

```text
Do not adopt tsfresh as the current mainline feature set.
Use manual_basic for downstream baseline planning.
```
