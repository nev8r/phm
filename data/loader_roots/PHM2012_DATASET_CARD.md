# PHM2012 / PRONOSTIA Dataset Card

更新日期：2026-06-18  
本地入口：`data/loader_roots/phm2012`

## 1. 数据集概览

| 项目 | 内容 |
|---|---|
| 数据集名称 | IEEE PHM 2012 Prognostic Challenge / PRONOSTIA bearing run-to-failure data |
| 数据类型 | 滚动轴承加速退化全寿命数据 |
| 主要传感器 | 水平、垂直方向加速度传感器；温度传感器 |
| 官方用途 | 轴承健康评估、诊断、预测和 RUL 估计方法验证 |
| 本地入口 | `data/loader_roots/phm2012` |
| 当前 loader | `PHM2012Loader` |

PRONOSTIA 是 FEMTO-ST 提出的轴承加速退化试验平台。PHM2012 challenge 使用该平台采集的轴承 run-to-failure 数据，目标是基于监测信号估计轴承剩余寿命。官方说明中，轴承状态监测包含温度信号和振动信号；振动由水平、垂直两个方向的加速度传感器采集。

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

含义是：时间 `09:39:39.065664` 时，水平振动加速度为 `0.552`，垂直振动加速度为 `-0.146`。当前 loader 只保留最后两列加速度信号。

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

本地副本约 `1.9 GB`，包含 `17` 个轴承实体、`24,889` 个 `acc_*.csv` 文件和 `3,018` 个 `temp_*.csv` 文件。

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

## 6. 当前框架 loader 返回形态

`PHM2012Loader` 负责注册目录、读取 `acc_*.csv`、取最后两列加速度信号、统一列名，并组装为 `Entity`。它不读取温度文件，不做特征分析，不生成训练标签。

```python
from phm.data.loader.PHM2012Loader import PHM2012Loader

loader = PHM2012Loader("data/loader_roots/phm2012")
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
#   "continuum": 2560,
#   "time_unit": "minute",
#   "span": 1 / 6,
#   "rul": 0,
#   "life": ...,
# }
```

字段说明：

| 字段 | 含义 |
|---|---|
| `Horizontal Vibration` | 水平方向振动加速度信号 |
| `Vertical Vibration` | 垂直方向振动加速度信号 |
| `frequency` | 加速度采样频率，当前为 `25600` |
| `continuum` | 单个加速度快照采样点数，当前为 `2560` |
| `span` | 单个 `acc_*.csv` 在 loader 中对应的时间跨度，当前为 `1/6` 分钟 |
| `life` | loader 根据样本长度换算的实体生命周期长度 |

## 7. 说明边界

- 本 card 只描述数据集、本地副本和当前 loader 输出形态。
- 原始大文件通过软链接接入，不建议提交到 Git。
- 若迁移机器，需要重新映射 `Learning_set` 和 `Full_Test_Set`。
- 后续标签构造、特征提取、标准化、训练集划分都不属于 loader 职责。

## 8. 来源

- PRONOSTIA platform paper: `https://publiweb.femto-st.fr/tntnet/entries/1528/documents/author/data`
- IEEE PHM 2012 challenge / PRONOSTIA related documentation.
