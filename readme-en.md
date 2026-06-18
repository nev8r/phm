<div align="center">
  <img src="image/phm-logo.png" alt="PHM" width="400">
</div>

<div align="center">
<h3>Bearing RUL Prediction and Fault Diagnosis System</h3>
</div>

<div align="center">

[![GPLv3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
</div>

<div align="center">

[简体中文](README.md) | [English](readme-en.md)

</div>

<div align="center">
    <a href="https://gitee.com/holdenmcgorin/phm" target="_blank">Gitee</a> •
    <a href="https://github.com/nev8r/phm" target="_blank">GitHub</a>
</div>

###  
> 1. **PHM** (Bearing PHM Framework) is designed for bearing Prognostics and Health Management (PHM) scenarios and supports deep-learning-based bearing tasks such as **Remaining Useful Life (RUL) prediction, fault diagnosis, and degradation stage analysis**.
> 2. The framework aims to provide a **unified and modular** research and experimentation platform for bearing studies. It standardizes data processing, model training, and performance evaluation, simplifying experiment construction and supporting comparison across bearing task types.


## 📦    Environment Management

This project uses `uv` to manage the Python 3.11 environment, dependencies, and lockfile. On macOS, the default sync installs the macOS PyTorch wheel from PyPI.

```bash
uv sync
```

## 🚀     Feature Overview
- ✅ **PyTorch-based Training Workflow**: Supports bearing RUL prediction and fault diagnosis experiments with PyTorch.

- 📦 **Automatic Bearing Dataset Import**: Built-in support for XJTU-SY and PHM2012 bearing datasets

- 📝 **Automatic Logging of Experimental Parameters and Results**: Includes model configs, regularization terms, iteration counts, sampling settings, etc.

- 🔁 **Custom Callback Support for Each Epoch**: Built-in EarlyStopping and TensorBoard are both implemented through callbacks.

- 🛠 **Model Training Monitoring**: Supports TensorBoard visualization and logging/alarming for gradient anomalies (vanishing/exploding gradients).

- 🔍 **Preprocessing & Feature Extraction**: Includes sliding window, normalization, RMS, kurtosis, and other techniques

- 🧠 **Flexible Degradation Stage Segmentation**: Supports 3σ rule, FPT (First Predictable Time), and more

- 🔮 **Versatile Prediction Methods**: Enables end-to-end forecasting, step-by-step rolling prediction, and uncertainty modeling

- 📊 **Rich Result Visualization**: Confusion matrices, degradation curves, prediction plots, attention maps, and more

- 📁 **Support for Multiple File Formats**: Easily import/export models, datasets, results, and caches in CSV, PKL, etc.

- 📈 **Comprehensive Evaluation Metrics**: MAE, MSE, RMSE, MAPE, PHM2012 Score, and more

- 🔧 **Modular and Extensible Design**: Add custom algorithms or components with minimal effort

## 💻    Unified CLI Example

The `phm` command provides one entry point for data analysis, paper reproduction training, baseline benchmarking, and run summaries:

```bash
uv run phm analyze --task all --full
uv run phm train --task rul --preset paper --full --device auto
uv run phm train --task fault --preset paper --full --device auto
uv run phm benchmark --task all --baselines all --full
```


## 📚 Reproduced Papers
> This framework has been used to reproduce several published methods in the bearing PHM field.  
> The original contributions are treated with full respect. If the reproduced results deviate from the originals, it may be due to differences in implementation or experimental setup, or potentially due to oversights during reproduction.

### ✅ Reproduced Papers Overview
- PHM2012 RUL: CBAM-CNN-LSTM with Hann-windowed rFFT features and degradation statistics.
- XJTU-SY Fault: ResCNN-LSTM with dual-channel time-frequency features for Healthy/Faulty classification.
- Benchmark: Ridge, RandomForest, sktime Rocket baselines, and deep models compared under shared splits and feature caches.

## 📂    File Structure
- src/USTC/SSE/BearingPrediction – Core framework code.
- doc – Detailed documentation (recommended for writing custom components).
- examples – Supporting examples and paper reproduction experiments.

### 📦 Dataset Sources

| Name              | Description                                                                 | Link                                                                 |
|-------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| XJTU-SY Dataset   | Rolling bearing degradation dataset published by Xi'an Jiaotong University | [Visit](https://biaowang.tech/xjtu-sy-bearing-datasets/)            |
| PHM2012 Dataset   | Bearing fault dataset from the IEEE PHM 2012 data challenge                 | [Visit](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) |


## ⚠     Important Notes
> - This framework is developed using Python 3.11. Compatibility issues may arise with other versions.
> - When reading datasets, do not change the internal file structure of the original datasets (you may keep only partial data). Altering the file structure may lead to data reading failures.
