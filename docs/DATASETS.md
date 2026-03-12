# 数据集说明

## 1. XJTU-SY

### 官方页面

`https://biaowang.tech/xjtu-sy-bearing-datasets/`

### 可尝试直链

`https://drive.google.com/uc?export=download&id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK`

### 数据集特点

- 西安交通大学发布
- 面向滚动轴承全寿命退化分析
- 适合 RUL 预测、退化阶段识别和健康指标建模

### 使用方式

```python
from USTC.SSE.BearingPrediction.api import XJTULoader

loader = XJTULoader("/path/to/XJTU-SY_Bearing_Datasets")
entity_ids = loader.list_entities()
entity = loader.load_entity("Bearing1_1")
```

### 目录要求

loader 会递归扫描目录，因此允许一定的目录差异。推荐组织方式：

```text
XJTU-SY_Bearing_Datasets
└── 35Hz12kN
    └── Bearing1_1
        ├── 1.csv
        ├── 2.csv
        └── ...
```

### 说明

XJTU-SY 下载入口可能切换镜像或出现 Google Drive 确认页，因此框架同时支持：

- 通过官方页面手动下载
- 通过直链尝试自动下载
- 直接加载已解压目录

## 2. PHM2012 / FEMTO

### 官方页面

NASA Data Portal：

`https://data.nasa.gov/dataset/FEMTO-Bearing-Dataset/jujd-xjyk`

### 直接下载地址

`https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip`

### 使用方式

```python
from USTC.SSE.BearingPrediction.api import PHM2012Loader

loader = PHM2012Loader("/path/to/FEMTO")
entity = loader.load_entity("Bearing1_1")
```

### 目录要求

```text
FEMTO
├── Learning_set
│   └── Bearing1_1
├── Test_set
│   └── Bearing1_4
└── Full_Test_Set
```

## 3. 合成数据

### 用途

- 开发时快速验证
- 单元测试
- 无真实数据时的示例实验

### 使用方式

```python
from USTC.SSE.BearingPrediction.api import SyntheticBearingFactory

factory = SyntheticBearingFactory(random_state=42)
entity = factory.create_run_to_failure_entity("Bearing1_1", snapshot_count=48, signal_length=2048)
```

## 4. 实体数据抽象

`BearingEntity` 统一封装不同数据集，核心字段包括：

- `entity_id`
- `dataset_name`
- `samples`
- `sample_rate`
- `metadata`

支持通道：

- `Horizontal Vibration`
- `Vertical Vibration`
