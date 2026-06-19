# PHM2012 / PRONOSTIA Dataset Card

更新日期：2026-06-19
本地入口：`data/loader_roots/phm2012`

## 1. 数据集概览

| 项目 | 内容 |
|---|---|
| 数据集名称 | IEEE PHM 2012 Prognostic Challenge / PRONOSTIA bearing run-to-failure data |
| 数据类型 | 滚动轴承加速退化全寿命数据 |
| 主要传感器 | 水平、垂直方向加速度传感器；温度传感器 |
| 官方用途 | 轴承健康评估、诊断、预测和 RUL 估计方法验证 |
| 当前项目入口 | `data/loader_roots/phm2012` |
| 当前 index 构建器 | `IndexBuilder._build_phm2012()` |
| 当前 raw reader | `RawSampleReader._read_phm2012()` |

PRONOSTIA 是 FEMTO-ST 提出的轴承加速退化试验平台。PHM2012 challenge 使用该平台采集的轴承 run-to-failure 数据，目标是基于监测信号估计轴承剩余寿命。官方说明中，轴承状态监测包含温度信号和振动信号；本项目当前只读取 `acc_*.csv` 中的水平、垂直振动加速度信号。

## 2. 官方采集信息

| 项目 | 内容 |
|---|---|
| 加速度采样频率 | 25.6 kHz |
| 温度采样频率 | 10 Hz |
| 振动方向 | Horizontal、Vertical |
| 振动文件常见列 | hour、minute、second、microsecond、horizontal acceleration、vertical acceleration |
| 本地 `acc_*.csv` 单文件行数 | 2560 行 |
| 本地 `acc_*.csv` 单文件覆盖时长 | 约 0.1 秒 |
| 本地相邻 `acc_*.csv` 间隔 | 约 10 秒 |

示例行：

```text
9,39,39,65664,0.552,-0.146
```

含义是：时间 `09:39:39.065664` 时，水平振动加速度为 `0.552`，垂直振动加速度为 `-0.146`。当前 `RawSampleReader` 只保留最后两列加速度信号。

读取后返回：

```python
signal.shape == (2560, 2)
channels == ["h", "v"]
```

其中 `h` 表示水平通道，`v` 表示垂直通道。reader 不读取温度文件，不做特征分析，不生成训练标签。

## 3. 工况与轴承

| 工况 | 轴承名前缀 | 转速 | 载荷 |
|---|---|---:|---:|
| Condition 1 | `Bearing1_*` | 1800 rpm | 4000 N |
| Condition 2 | `Bearing2_*` | 1650 rpm | 4200 N |
| Condition 3 | `Bearing3_*` | 1500 rpm | 5000 N |

## 4. 本地目录

本地 `phm2012` 目录下的两个子目录是软链接，指向仓库外部真实数据目录：

```text
data/loader_roots/phm2012/
├── Learning_set -> external Training_set/Learning_set
│   ├── Bearing1_1/
│   ├── Bearing1_2/
│   ├── Bearing2_1/
│   ├── Bearing2_2/
│   ├── Bearing3_1/
│   └── Bearing3_2/
└── Full_Test_Set -> external Validation_Set/Full_Test_Set
    ├── Bearing1_3/
    ├── Bearing1_4/
    ├── Bearing1_5/
    ├── Bearing1_6/
    ├── Bearing1_7/
    ├── Bearing2_3/
    ├── Bearing2_4/
    ├── Bearing2_5/
    ├── Bearing2_6/
    ├── Bearing2_7/
    └── Bearing3_3/
```

当前本地索引统计为 `17` 个轴承实体、`24,889` 个 `acc_*.csv` sample。`temp_*.csv` 温度文件存在于部分轴承目录中，但当前 Stage 0+ 主线没有读取温度列。

## 5. 本地轴承清单

| 子集 | 轴承 | `acc_*.csv` | `temp_*.csv` |
|---|---|---:|---:|
| Learning_set | Bearing1_1 | 2803 | 466 |
| Learning_set | Bearing1_2 | 871 | 144 |
| Learning_set | Bearing2_1 | 911 | 151 |
| Learning_set | Bearing2_2 | 797 | 0 |
| Learning_set | Bearing3_1 | 515 | 89 |
| Learning_set | Bearing3_2 | 1637 | 0 |
| Full_Test_Set | Bearing1_3 | 2375 | 0 |
| Full_Test_Set | Bearing1_4 | 1428 | 237 |
| Full_Test_Set | Bearing1_5 | 2463 | 410 |
| Full_Test_Set | Bearing1_6 | 2448 | 408 |
| Full_Test_Set | Bearing1_7 | 2259 | 376 |
| Full_Test_Set | Bearing2_3 | 1955 | 0 |
| Full_Test_Set | Bearing2_4 | 751 | 125 |
| Full_Test_Set | Bearing2_5 | 2311 | 386 |
| Full_Test_Set | Bearing2_6 | 701 | 116 |
| Full_Test_Set | Bearing2_7 | 230 | 38 |
| Full_Test_Set | Bearing3_3 | 434 | 72 |

## 6. 当前框架 index 后的样子

`IndexBuilder` 会为每个 `acc_*.csv` 生成一行 sample index。核心字段如下：

| 字段 | 含义 | 示例 |
|---|---|---|
| `sample_uid` | 项目内唯一 sample 标识 | `PHM2012::Bearing1_1::000001` |
| `dataset` | 数据集名 | `PHM2012` |
| `bearing_id` | 轴承编号 | `Bearing1_1` |
| `condition_id` | 工况 | `Condition1` |
| `source_group` | 官方子集 | `Learning_set` 或 `Full_Test_Set` |
| `sample_id` | 从文件名提取并补齐的 sample 编号 | `000001` |
| `timestep` | 在该轴承内部按文件顺序得到的时间步 | `0` |
| `file_path` | 原始 CSV 路径 | `data/loader_roots/phm2012/Learning_set/Bearing1_1/acc_00001.csv` |
| `sampling_rate` | 加速度采样频率 | `25600` |
| `expected_points` | 单个 `acc` 快照点数 | `2560` |
| `sample_interval_seconds` | 相邻 `acc` 快照间隔 | `10` |
| `channel_names` | 通道名 | `Horizontal Vibration,Vertical Vibration` |

实际读取一个 sample 时：

```python
from USTC.SSE.BearingPrediction.infra.feature.RawSampleReader import RawSampleReader

signal, channels = RawSampleReader().read(sample_row)
```

返回：

```text
signal: numpy.ndarray, shape = (2560, 2)
channels: ["h", "v"]
```

## 7. 说明边界

- 本 card 只描述数据集、本地副本和当前框架读取后的形态。
- 当前项目不把温度文件纳入 Stage 0+ 主线。
- 特征提取、标准化、标签构造、任务窗口构造不属于 raw reader 职责。
- 当前项目固定划分见 [loading-and-splits.md](loading-and-splits.md)。
- 原始大文件通过软链接接入，不建议提交到 Git。

## 8. 来源

- PRONOSTIA platform paper: `https://publiweb.femto-st.fr/tntnet/entries/1528/documents/author/data`
- IEEE PHM 2012 challenge / PRONOSTIA related documentation.
