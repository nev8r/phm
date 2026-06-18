# XJTU-SY Dataset Card

更新日期：2026-06-18  
本地入口：`data/loader_roots/xjtu`

## 1. 数据集概览

| 项目 | 内容 |
|---|---|
| 数据集名称 | XJTU-SY Bearing Datasets |
| 发布方 | Xi'an Jiaotong University (XJTU) 与 Changxing Sumyoung Technology Co., Ltd. (SY) |
| 数据类型 | 滚动轴承加速退化全寿命数据 |
| 轴承数量 | 15 个轴承，3 个工况，每个工况 5 个轴承 |
| 主要传感器 | 水平、垂直方向加速度传感器 |
| 官方用途 | 滚动轴承预测、健康评估和 RUL 方法验证 |
| 本地入口 | `data/loader_roots/xjtu` |
| 当前 loader | `XJTULoader` |

XJTU-SY 数据集包含 15 个滚动轴承的 run-to-failure 振动数据。官方说明中，每次采样保存为一个 CSV 文件，第一列是水平振动信号，第二列是垂直振动信号；同时官方明细表给出每个轴承的 CSV 文件数、寿命和 fault element。

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

本地 CSV 表头为：

```text
Horizontal_vibration_signals,Vertical_vibration_signals
```

当前 loader 将其标准化为：

```text
Horizontal Vibration
Vertical Vibration
```

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

本地副本约 `11 GB`，包含 `9,216` 个 CSV 文件。

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

这里的 fault element 是轴承级说明，用于描述轴承最终或主要故障元素。原始 CSV 文件本身不包含逐行或逐窗口故障标签。当前 `XJTULoader` 将这些轴承级 fault element 映射为项目内的 `Fault` 枚举。

## 6. 当前框架 loader 返回形态

`XJTULoader` 负责注册目录、按数字文件名排序读取 CSV、统一列名，并组装为 `Entity`。它不做特征分析，也不生成窗口级 Healthy/Faulty 标签。

```python
from phm.data.loader.XJTULoader import XJTULoader

loader = XJTULoader("data/loader_roots/xjtu")
bearing = loader.load_entity("Bearing1_1")
```

返回对象：

```python
bearing.name
# "Bearing1_1"

bearing.data
# {
#   "Horizontal Vibration": ndarray(N, 1),
#   "Vertical Vibration": ndarray(N, 1),
# }

bearing.meta
# {
#   "frequency": 25600,
#   "continuum": 32768,
#   "time_unit": "minute",
#   "span": 1,
#   "rul": 0,
#   "life": ...,
#   "fault_type": [Fault.OF],
# }
```

字段说明：

| 字段 | 含义 |
|---|---|
| `Horizontal Vibration` | 水平方向振动加速度信号 |
| `Vertical Vibration` | 垂直方向振动加速度信号 |
| `frequency` | 加速度采样频率，当前为 `25600` |
| `continuum` | 单个 CSV 快照采样点数，当前为 `32768` |
| `span` | 单个 CSV 在 loader 中对应的时间跨度，当前为 `1` 分钟 |
| `fault_type` | 由轴承级 fault element 映射得到的项目内枚举列表 |

## 7. 说明边界

- 本 card 只描述数据集、本地副本和当前 loader 输出形态。
- 原始大文件通过软链接接入，不建议提交到 Git。
- 若迁移机器，需要重新映射 `data/loader_roots/xjtu`。
- 后续标签构造、特征提取、标准化、训练集划分都不属于 loader 职责。

## 8. 来源

- XJTU-SY official page: `https://biaowang.tech/xjtu-sy-bearing-datasets/`
- Cited paper: Biao Wang, Yaguo Lei, Naipeng Li, Ningbo Li, "A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings", IEEE Transactions on Reliability, 2020.
