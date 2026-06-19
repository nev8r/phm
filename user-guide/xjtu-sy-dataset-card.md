# XJTU-SY Dataset Card

更新日期：2026-06-19
本地入口：`data/loader_roots/xjtu`

## 1. 数据集概览

| 项目 | 内容 |
|---|---|
| 数据集名称 | XJTU-SY Bearing Datasets |
| 发布方 | Xi'an Jiaotong University (XJTU) 与 Changxing Sumyoung Technology Co., Ltd. (SY) |
| 数据类型 | 滚动轴承加速退化全寿命振动数据 |
| 轴承数量 | 15 个轴承，3 个工况，每个工况 5 个轴承 |
| 主要传感器 | 水平、垂直方向加速度传感器 |
| 官方用途 | 滚动轴承预测、健康评估和 RUL 方法验证 |
| 当前项目入口 | `data/loader_roots/xjtu` |
| 当前 index 构建器 | `IndexBuilder._build_xjtu_sy()` |
| 当前 raw reader | `RawSampleReader._read_xjtu()` |

XJTU-SY 包含 15 个滚动轴承的 run-to-failure 数据。每个 CSV 文件表示一次振动快照，项目内把每个 CSV 视为一个 sample。CSV 中包含水平与垂直两个方向的振动加速度信号。

## 2. 官方采集信息

| 项目 | 内容 |
|---|---|
| 加速度传感器 | PCB 352C33，两只传感器相隔 90 度安装 |
| 采样频率 | 25.6 kHz |
| 单次采样点数 | 32768 点 |
| 单次采样覆盖时长 | 1.28 秒 |
| 采样周期 | 1 分钟 |
| CSV 第 1 列 | Horizontal vibration signal |
| CSV 第 2 列 | Vertical vibration signal |

本地 CSV 表头通常为：

```text
Horizontal_vibration_signals,Vertical_vibration_signals
```

当前 `RawSampleReader` 读取后返回：

```python
signal.shape == (32768, 2)
channels == ["h", "v"]
```

其中 `h` 表示水平通道，`v` 表示垂直通道。reader 只负责读取原始两通道数值，不做特征分析、不做窗口构造、不生成标签。

## 3. 工况

| 工况目录 | 转速 | 载荷 | 轴承 |
|---|---:|---:|---|
| `35Hz12kN` | 35 Hz，约 2100 rpm | 12 kN | `Bearing1_1` - `Bearing1_5` |
| `37.5Hz11kN` | 37.5 Hz，约 2250 rpm | 11 kN | `Bearing2_1` - `Bearing2_5` |
| `40Hz10kN` | 40 Hz，约 2400 rpm | 10 kN | `Bearing3_1` - `Bearing3_5` |

## 4. 本地目录

本地 `xjtu` 是软链接，指向仓库外部真实数据目录：

```text
data/loader_roots/xjtu -> external XJTU-SY_Bearing_Datasets
├── 35Hz12kN/
│   ├── Bearing1_1/
│   ├── Bearing1_2/
│   ├── Bearing1_3/
│   ├── Bearing1_4/
│   └── Bearing1_5/
├── 37.5Hz11kN/
│   ├── Bearing2_1/
│   ├── Bearing2_2/
│   ├── Bearing2_3/
│   ├── Bearing2_4/
│   └── Bearing2_5/
└── 40Hz10kN/
    ├── Bearing3_1/
    ├── Bearing3_2/
    ├── Bearing3_3/
    ├── Bearing3_4/
    └── Bearing3_5/
```

当前本地索引统计为 `9,216` 个 CSV sample。

## 5. 本地轴承清单

| 工况 | 轴承 | CSV 文件数 | 轴承级 fault element |
|---|---|---:|---|
| 35Hz12kN | Bearing1_1 | 123 | Outer Race Fault |
| 35Hz12kN | Bearing1_2 | 161 | Outer Race Fault |
| 35Hz12kN | Bearing1_3 | 158 | Outer Race Fault |
| 35Hz12kN | Bearing1_4 | 122 | Cage Fault |
| 35Hz12kN | Bearing1_5 | 52 | Inner Race Fault + Outer Race Fault |
| 37.5Hz11kN | Bearing2_1 | 491 | Inner Race Fault |
| 37.5Hz11kN | Bearing2_2 | 161 | Outer Race Fault |
| 37.5Hz11kN | Bearing2_3 | 533 | Cage Fault |
| 37.5Hz11kN | Bearing2_4 | 42 | Outer Race Fault |
| 37.5Hz11kN | Bearing2_5 | 339 | Outer Race Fault |
| 40Hz10kN | Bearing3_1 | 2538 | Outer Race Fault |
| 40Hz10kN | Bearing3_2 | 2496 | Inner Race Fault + Outer Race Fault + Cage Fault + Ball Fault |
| 40Hz10kN | Bearing3_3 | 371 | Inner Race Fault |
| 40Hz10kN | Bearing3_4 | 1515 | Inner Race Fault |
| 40Hz10kN | Bearing3_5 | 114 | Outer Race Fault |

这里的 fault element 是轴承级说明，用于描述轴承最终或主要故障元素。原始 CSV 文件本身不包含逐行、逐窗口故障标签。

## 6. 当前框架 index 后的样子

`IndexBuilder` 会为每个 CSV 生成一行 sample index。核心字段如下：

| 字段 | 含义 | 示例 |
|---|---|---|
| `sample_uid` | 项目内唯一 sample 标识 | `XJTU-SY::Bearing1_1::000001` |
| `dataset` | 数据集名 | `XJTU-SY` |
| `bearing_id` | 轴承编号 | `Bearing1_1` |
| `condition_id` | 工况目录 | `35Hz12kN` |
| `source_group` | XJTU-SY 无官方 Learning/Test 组，当前为 `None` | `None` |
| `sample_id` | 从文件名提取并补齐的 sample 编号 | `000001` |
| `timestep` | 在该轴承内部按文件顺序得到的时间步 | `0` |
| `file_path` | 原始 CSV 路径 | `data/loader_roots/xjtu/35Hz12kN/Bearing1_1/1.csv` |
| `sampling_rate` | 采样频率 | `25600` |
| `expected_points` | 单次快照点数 | `32768` |
| `sample_interval_seconds` | 相邻快照间隔 | `60` |
| `channel_names` | 通道名 | `Horizontal Vibration,Vertical Vibration` |

实际读取一个 sample 时：

```python
from USTC.SSE.BearingPrediction.infra.feature.RawSampleReader import RawSampleReader

signal, channels = RawSampleReader().read(sample_row)
```

返回：

```text
signal: numpy.ndarray, shape = (32768, 2)
channels: ["h", "v"]
```

## 7. 说明边界

- 本 card 只描述数据集、本地副本和当前框架读取后的形态。
- 特征提取、标准化、标签构造、任务窗口构造不属于 raw reader 职责。
- 当前项目固定划分见 [loading-and-splits.md](loading-and-splits.md)。
- 原始大文件通过软链接接入，不建议提交到 Git。

## 8. 来源

- XJTU-SY official page: `https://biaowang.tech/xjtu-sy-bearing-datasets/`
- Cited paper: Biao Wang, Yaguo Lei, Naipeng Li, Ningbo Li, "A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings", IEEE Transactions on Reliability, 2020.
