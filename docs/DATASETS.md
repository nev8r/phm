# 数据集说明

本文档面向第一次接触本项目数据的人，重点说明本仓库当前支持的两个真实轴承寿命数据集：

- `XJTU-SY`：西安交通大学与昇阳科技发布的滚动轴承全寿命退化数据集。
- `PHM2012 / FEMTO / PRONOSTIA`：FEMTO-ST PRONOSTIA 平台产生，并用于 2012 IEEE PHM Prognostic Challenge 的轴承寿命数据集。

本项目的主线是 **RUL 预测、失效概率建模与生存分析**。两个数据集都更适合做剩余寿命预测、健康指标构造、退化阶段划分和生存概率分析，不应把它们简单理解为故障类型分类数据集。

## 1. 本项目的数据抽象

所有真实数据集 loader 最终都会返回 `BearingEntity`。这样后续预处理、特征提取、训练和可视化可以使用统一接口。

核心字段如下。

| 字段 | 含义 |
| --- | --- |
| `entity_id` | 轴承编号，例如 `Bearing1_1` |
| `dataset_name` | 数据集名称，例如 `XJTU-SY` 或 `PHM2012` |
| `samples` | 按时间排序的采样快照表 |
| `sample_rate` | 振动信号采样率，本项目中两个 loader 均按 `25600.0` Hz 处理 |
| `metadata` | 数据集来源、工况、采样周期、RUL 单位等说明 |

`samples` 中常用列如下。

| 列名 | 含义 |
| --- | --- |
| `sample_index` | 从 0 开始的快照序号 |
| `timestamp` | 快照对应的秒级时间戳；XJTU-SY 从 0 开始，PHM2012 从文件时间列解析 |
| `elapsed_seconds` | 相对第一条快照的运行时间，单位秒 |
| `rul` | 剩余寿命标签，当前 loader 统一使用秒作为单位 |
| `Horizontal Vibration` | 水平方向振动信号数组 |
| `Vertical Vibration` | 垂直方向振动信号数组 |
| `Temperature` | PHM2012 温度信号数组；XJTU-SY 没有该列 |
| `source_file` | 对应的原始振动文件名 |
| `temperature_file` | PHM2012 对应温度文件名；没有匹配文件时为 `None` |

`metadata` 中常用键如下。

| 键 | 含义 |
| --- | --- |
| `entity_path` | loader 实际解析到的轴承目录 |
| `sample_rate_hz` | 振动采样率，通常为 `25600.0` |
| `sampling_period_seconds` | 两个相邻快照之间的采样间隔 |
| `snapshot_duration_seconds` | 单个快照覆盖的真实采样时长 |
| `rul_unit` | `rul` 的单位，当前为 `seconds` |
| `operating_condition` | 工况说明 |
| `rotating_speed_rpm` | 转速，单位 rpm |
| `radial_load_kn` / `radial_load_n` | 径向载荷，分别对应 XJTU-SY 和 PHM2012 |

## 2. XJTU-SY 数据集

### 2.1 数据集来源与用途

`XJTU-SY` 是滚动轴承加速寿命试验数据集，包含 15 个轴承在 3 种工况下的 run-to-failure 振动数据。数据集适合验证 prognostics 算法，尤其适合：

- RUL 回归预测；
- 健康指标构造；
- 退化趋势分析；
- 退化阶段划分；
- 跨工况泛化实验。

官方介绍页：

- `https://biaowang.tech/xjtu-sy-bearing-datasets/`

项目中的 loader：

```python
from USTC.SSE.BearingPrediction.api import XJTULoader

loader = XJTULoader("/path/to/XJTU-SY_Bearing_Datasets")
entity_ids = loader.list_entities()
entity = loader.load_entity("Bearing1_1")
```

### 2.2 工况与轴承编号

XJTU-SY 共有 3 个典型工况，每个工况 5 个轴承。

| 目录名 | 转速 | 径向载荷 | 常见轴承编号 |
| --- | --- | --- | --- |
| `35Hz12kN` | `2100 rpm` / `35 Hz` | `12 kN` | `Bearing1_1` 至 `Bearing1_5` |
| `37.5Hz11kN` | `2250 rpm` / `37.5 Hz` | `11 kN` | `Bearing2_1` 至 `Bearing2_5` |
| `40Hz10kN` | `2400 rpm` / `40 Hz` | `10 kN` | `Bearing3_1` 至 `Bearing3_5` |

本项目会从目录名解析工况信息。例如 `37.5Hz11kN/Bearing2_1` 会在 `metadata` 中得到：

```python
{
    "operating_condition": "37.5Hz11kN",
    "rotating_speed_hz": 37.5,
    "rotating_speed_rpm": 2250.0,
    "radial_load_kn": 11.0,
}
```

### 2.3 推荐目录结构

把解压后的数据放在 `data/raw/XJTU-SY_Bearing_Datasets` 下最清晰：

```text
data/raw/XJTU-SY_Bearing_Datasets
├── 35Hz12kN
│   ├── Bearing1_1
│   │   ├── 1.csv
│   │   ├── 2.csv
│   │   └── ...
│   └── Bearing1_2
├── 37.5Hz11kN
│   └── Bearing2_1
└── 40Hz10kN
    └── Bearing3_1
```

`XJTULoader` 会递归扫描目录，所以顶层目录名可以不同，但每个轴承目录最好保持 `Bearingx_y` 形式。

### 2.4 CSV 文件格式

每个 CSV 文件代表一次采样快照。

| 列 | 含义 | 本项目映射 |
| --- | --- | --- |
| 第 1 列 | 水平方向振动信号 | `Horizontal Vibration` |
| 第 2 列 | 垂直方向振动信号 | `Vertical Vibration` |

loader 同时兼容无表头 CSV 和带表头 CSV。例如以下两种都可以解析：

```text
0.012,0.021
0.013,0.020
```

```text
Horizontal_vibration_signals,Vertical_vibration_signals
0.012,0.021
0.013,0.020
```

### 2.5 采样参数与时间语义

XJTU-SY 的关键采样参数如下。

| 参数 | 数值 |
| --- | --- |
| 振动采样率 | `25.6 kHz` |
| 单个 CSV 采样点数 | `32768` |
| 单个 CSV 覆盖时长 | `1.28 s` |
| 相邻 CSV 采样间隔 | `1 min` |
| 通道数 | 2 个振动通道 |

这意味着：

- `1.csv` 不是运行第 1 秒的数据，而是第 1 次快照；
- `2.csv` 通常表示 1 分钟后的下一次快照；
- 每次快照内部有 `32768` 个高频点，覆盖 `1.28 s` 的短时间片段；
- 快照之间的中间时间没有连续原始振动数据。

本项目中 `XJTULoader` 的时间列含义如下。

| 列 | 计算方式 |
| --- | --- |
| `timestamp` | `sample_index * 60.0` |
| `elapsed_seconds` | 相对第一条样本的秒数 |
| `rul` | `(最后一个 sample_index - 当前 sample_index) * 60.0` |
| `rul_unit` | `seconds` |

示例：

```python
entity = XJTULoader("data/raw/XJTU-SY_Bearing_Datasets").load_entity("Bearing1_1")
entity.samples[["sample_index", "timestamp", "elapsed_seconds", "rul", "source_file"]].head()
```

### 2.6 适合的实验任务

XJTU-SY 更适合以下实验：

- 单轴承时间顺序训练和测试；
- 同工况下多轴承训练、留一个轴承测试；
- 跨工况迁移，例如用 `35Hz12kN` 训练，用 `37.5Hz11kN` 测试；
- 原始信号输入的 CNN、RNN、Transformer；
- 基于统计特征的 MLP、XGBoost、随机森林等模型。

### 2.7 推荐实验切分

对于课程项目，推荐先用简单、可解释的切分方式。

| 目标 | 推荐切分 |
| --- | --- |
| 快速跑通流程 | 单个轴承按时间 `70% / 30%` 切分 |
| 同工况泛化 | 同一工况下 4 个轴承训练，1 个轴承测试 |
| 跨工况泛化 | 一个或两个工况训练，另一个工况测试 |
| 报告展示 | 固定一个代表性轴承画 RUL 曲线，另选多轴承汇总指标 |

注意：如果按窗口随机划分，同一轴承相邻时间窗口可能同时出现在训练集和测试集，会造成数据泄漏。严谨实验应优先按轴承编号或时间顺序切分。

## 3. PHM2012 / FEMTO / PRONOSTIA 数据集

### 3.1 数据集来源与用途

`PHM2012` 是 2012 IEEE PHM Prognostic Challenge 使用的数据集，来自 FEMTO-ST 的 `PRONOSTIA` 轴承加速退化平台。它常被称为：

- `PHM2012`
- `FEMTO`
- `PRONOSTIA`
- `FEMTO-ST Bearing Dataset`

本项目统一使用 `PHM2012Loader` 读取该数据集。

参考入口：

- NASA Data Portal：`https://data.nasa.gov/dataset/FEMTO-Bearing-Dataset/jujd-xjyk`
- 直接压缩包：`https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip`
- PRONOSTIA 论文：`https://hal.science/hal-00719503`

使用方式：

```python
from USTC.SSE.BearingPrediction.api import PHM2012Loader

loader = PHM2012Loader("/path/to/FEMTO")
entity_ids = loader.list_entities()
entity = loader.load_entity("Bearing1_1")
```

### 3.2 工况与轴承编号

PHM2012 按 `Bearingx_y` 的第一个数字表示工况。

| 工况 | 转速 | 径向载荷 | 示例轴承 |
| --- | --- | --- | --- |
| Condition 1 | `1800 rpm` | `4000 N` | `Bearing1_1`, `Bearing1_2`, `Bearing1_3` |
| Condition 2 | `1650 rpm` | `4200 N` | `Bearing2_1`, `Bearing2_2`, `Bearing2_3` |
| Condition 3 | `1500 rpm` | `5000 N` | `Bearing3_1`, `Bearing3_2`, `Bearing3_3` |

本项目会根据 `Bearing1_1` / `Bearing2_1` / `Bearing3_1` 这样的编号推断：

```python
{
    "operating_condition": "Condition 1",
    "rotating_speed_rpm": 1800,
    "radial_load_n": 4000,
}
```

### 3.3 推荐目录结构

PHM2012 解压后常见目录包括 `Learning_set`、`Test_set`、`Full_Test_Set` 等。推荐放在：

```text
data/raw/FEMTO
├── Training_set
│   └── Learning_set
│       ├── Bearing1_1
│       │   ├── acc_00001.csv
│       │   ├── temp_00001.csv
│       │   ├── acc_00002.csv
│       │   └── temp_00002.csv
│       └── Bearing1_2
├── Test_set
│   ├── Bearing1_3
│   └── Bearing2_4
└── Full_Test_Set
```

本项目的 `PHM2012Loader` 会递归寻找 `Bearingx_y` 目录，不强依赖最外层目录名，但建议保留原始目录层级，便于判断 `split_name`。

### 3.4 `acc_00001.csv` 振动文件格式

PHM2012 的加速度文件通常命名为 `acc_00001.csv`、`acc_00002.csv` 等。本项目只把 `acc_*.csv` 作为主时间线，避免温度文件被误当成振动快照。

常见列含义如下。

| 列序号 | 含义 | 本项目映射 |
| --- | --- | --- |
| 第 1 列 | hour | 用于解析 `timestamp` |
| 第 2 列 | minute | 用于解析 `timestamp` |
| 第 3 列 | second | 用于解析 `timestamp` |
| 第 4 列 | microsecond / subsecond | 用于解析 `timestamp` |
| 第 5 列 | horizontal acceleration | `Horizontal Vibration` |
| 第 6 列 | vertical acceleration | `Vertical Vibration` |

文件可能使用分号或逗号分隔，`PHM2012Loader` 会自动尝试 `,`、`;` 和空白分隔。

### 3.5 `temp_00001.csv` 温度文件格式

温度文件通常命名为 `temp_00001.csv`、`temp_00002.csv` 等。loader 会按编号对齐同名快照：

- `acc_00001.csv` 对齐 `temp_00001.csv`
- `acc_00002.csv` 对齐 `temp_00002.csv`

温度文件最后一列会被读入 `Temperature` 通道。示例：

```python
entity = PHM2012Loader("data/raw/FEMTO").load_entity("Bearing1_1")
entity.samples[["source_file", "temperature_file"]].head()
entity.get_channel("Temperature")[0]
```

### 3.6 采样参数与时间语义

PHM2012 的关键采样参数如下。

| 参数 | 数值 |
| --- | --- |
| 振动采样率 | `25.6 kHz` |
| 单个振动文件采样点数 | `2560` |
| 单个振动文件覆盖时长 | `0.1 s` |
| 相邻振动文件采样间隔 | `10 s` |
| 温度采样率 | `10 Hz` |
| 主要通道 | 水平振动、垂直振动、温度 |

这意味着 PHM2012 与 XJTU-SY 很像，都是“高频短快照 + 较长间隔”的采样方式，但它们的间隔不同：

- XJTU-SY：每 `1 min` 取一次 `1.28 s` 快照；
- PHM2012：每 `10 s` 取一次 `0.1 s` 快照。

本项目中 `PHM2012Loader` 的时间列含义如下。

| 列 | 计算方式 |
| --- | --- |
| `timestamp` | 从 `acc_*.csv` 前 4 列解析为秒 |
| `elapsed_seconds` | 相对第一条样本的秒数 |
| `rul` | Learning_set 中按最后一个快照倒推，Test_set 中如有官方剩余寿命则叠加 |
| `rul_unit` | `seconds` |

### 3.7 Learning_set、Test_set 与 RUL 标签

PHM2012 和 XJTU-SY 的一个重要差异是：PHM2012 challenge 包含训练用的完整寿命数据和测试用的截断数据。

| split | 含义 | RUL 处理 |
| --- | --- | --- |
| `Learning_set` | 用于建模的完整 run-to-failure 数据 | 可从最后一个快照倒推出相对 RUL |
| `Test_set` | 截断观测数据，比赛要求预测截断点之后的剩余寿命 | loader 对已知轴承提供 `known_terminal_rul_seconds` |
| `Full_Test_Set` | 某些镜像中包含更完整的测试记录 | 应按实际目录和实验目标确认 |

本项目的 `PHM2012Loader` 已内置部分 challenge 测试轴承的终止 RUL 秒数，例如：

```python
entity = PHM2012Loader("data/raw/FEMTO").load_entity("Bearing1_3")
entity.metadata.get("known_terminal_rul_seconds")
```

如果只做课程展示，建议优先从 `Learning_set` 跑通完整流程，再在 `Test_set` 上做补充实验。

### 3.8 适合的实验任务

PHM2012 更适合以下任务：

- RUL 回归预测；
- 生存分析和失效概率估计；
- 带温度通道的多源特征分析；
- 同工况内少样本训练和测试；
- 使用 `C-index`、Brier Score 等生存分析指标。

它不适合直接做故障部位分类，因为 challenge 设定中并没有给出每条样本的明确故障类型标签。

### 3.9 推荐实验切分

| 目标 | 推荐切分 |
| --- | --- |
| 快速跑通流程 | `Learning_set/Bearing1_1` 按时间 `70% / 30%` 切分 |
| 同工况验证 | Condition 1 用 `Bearing1_1`, `Bearing1_2` 训练，用 `Bearing1_3` 等测试 |
| 生存分析 | 使用多个 bearing 的 `duration`、`event` 和特征表训练 |
| 报告展示 | 展示 `rul` 曲线、失效概率曲线、生存概率曲线和核心指标 |

注意：PHM2012 的训练轴承数量较少，过度复杂模型容易过拟合。课程项目中建议先用 MLP、XGBoost 或小型 CNN 跑通，再尝试 Transformer。

## 4. 两个数据集的核心差异

| 维度 | XJTU-SY | PHM2012 / FEMTO / PRONOSTIA |
| --- | --- | --- |
| 主要用途 | 轴承全寿命退化与 RUL 预测 | PHM challenge RUL 预测与生存分析 |
| 工况数 | 3 | 3 |
| 轴承数量 | 15 个 | 训练集 6 个完整寿命轴承，测试集若干截断轴承 |
| 振动采样率 | `25.6 kHz` | `25.6 kHz` |
| 单快照点数 | `32768` | `2560` |
| 单快照时长 | `1.28 s` | `0.1 s` |
| 快照间隔 | `1 min` | `10 s` |
| 主要文件 | `1.csv`, `2.csv`, ... | `acc_00001.csv`, `temp_00001.csv`, ... |
| 主要通道 | 水平振动、垂直振动 | 水平振动、垂直振动、温度 |
| 当前 loader | `XJTULoader` | `PHM2012Loader` |

## 5. 可运行示例

本仓库的 `examples/` 目录提供了可直接运行的 notebook 示例。为了避免依赖大型真实数据，每个 notebook 都会通过包内 helper 自动生成极小的数据集目录 fixture，再通过真实 loader 读取。

```bash
uv run --extra dev pytest tests/test_examples_notebooks.py -q
```

也可以在 Jupyter 或 VS Code 中打开任意 notebook，按顺序执行全部单元格。

当前 notebook 清单：

```text
examples/00_generate_demo_datasets.ipynb
examples/01_xjtu_loader_overview.ipynb
examples/02_phm2012_loader_overview.ipynb
examples/03_xjtu_cnn_rul_training.ipynb
examples/04_phm2012_mlp_feature_training.ipynb
examples/05_cross_dataset_feature_export.ipynb
```

这些 notebook 默认使用自动生成的小型 demo 数据，因此克隆仓库后即可执行。若后续要改成真实数据实验，可以保留 notebook 主体流程，把数据根目录替换为 `data/raw/XJTU-SY_Bearing_Datasets` 或 `data/raw/FEMTO`。

## 6. 常见问题

### 6.1 为什么 `rul` 是秒，不是文件编号？

两个数据集的文件编号不是同一时间单位。XJTU-SY 相邻文件间隔是 `1 min`，PHM2012 相邻文件间隔是 `10 s`。为了让模型目标有一致物理含义，本项目 loader 把 `rul` 统一转成秒，并在 `metadata["rul_unit"]` 中写明 `seconds`。

### 6.2 为什么 PHM2012 的 `timestamp` 不是从 0 开始？

PHM2012 的 `acc_*.csv` 文件前几列包含原始采集时间。`PHM2012Loader` 会先解析这些时间列作为 `timestamp`，再额外提供从 0 开始的 `elapsed_seconds`。

### 6.3 为什么 XJTU-SY 没有 `Temperature`？

当前 XJTU-SY loader 只解析两个振动通道：`Horizontal Vibration` 和 `Vertical Vibration`。PHM2012 才有可对齐的温度文件。

### 6.4 为什么不直接随机打散所有窗口？

随机打散窗口容易造成数据泄漏。比如同一轴承相邻一分钟的窗口非常相似，如果一部分进入训练集、一部分进入测试集，测试指标会过于乐观。正式实验建议按轴承或时间顺序切分。

### 6.5 如果真实数据目录和文档不完全一致怎么办？

两个 loader 都使用递归扫描，能容忍一定目录差异。最重要的是保留轴承目录名和文件命名规则：

- XJTU-SY：`Bearingx_y` 目录下有数字命名的 `.csv`；
- PHM2012：`Bearingx_y` 目录下有 `acc_*.csv`，可选 `temp_*.csv`。

### 6.6 两个数据集是否能用于故障诊断分类？

本项目不把这两个数据集作为故障诊断分类数据集使用。它们在本项目中的定位是 RUL 预测、退化阶段分析、失效概率建模和生存分析。若未来要做故障类型分类，需要额外引入明确标注故障类型的数据集或标签规则。

## 7. 数据管理建议

真实原始数据通常较大，不应提交到 git。建议放在：

```text
data/raw/XJTU-SY_Bearing_Datasets
data/raw/FEMTO
```

本仓库已忽略 `data/raw/**`，只保留 `data/raw/.gitkeep`。示例脚本生成的小数据和训练输出默认写入 `outputs/examples/`，也不会作为源码提交。
