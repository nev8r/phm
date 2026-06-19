# 数据加载与划分说明

更新日期：2026-06-19

本文档说明当前项目如何加载 XJTU-SY 与 PHM2012，并记录当前固定 train/val/test 划分。两个数据集本身的详细说明分别见：

- [xjtu-sy-dataset-card.md](xjtu-sy-dataset-card.md)
- [phm2012-dataset-card.md](phm2012-dataset-card.md)

## 1. 当前架构是否支持这种划分

支持。当前 Stage 0+ 主线按 sample index 划分，不依赖手工拆分后的目录树。

```text
dataset.root
→ IndexBuilder
→ sample_index.parquet / sample_index.csv
→ SplitRegistry
→ SplitResult(train_sample_uids, val_sample_uids, test_sample_uids)
→ FeatureExtractor / LabelBuilder / TaskBuilder
```

关键点：

- `IndexBuilder` 先扫描原始数据 root，为每个 CSV 快照生成唯一 `sample_uid`。
- splitter 不移动原始文件，只返回 train/val/test 三组 `sample_uid`。
- `FeatureExtractor` 从 index 的 `file_path` 读取原始 CSV。
- `FeatureCleaner` 有 split 时只用 train samples 估计标准化参数。
- `WindowBuilder` 在构造 TaskDataset 时通过 `sample_uid -> split` 映射继承同一份划分。

因此，当前划分方式会贯穿 index、feature、label、task、train/eval 全链路。

## 2. 本地数据 root

```text
data/loader_roots/xjtu
data/loader_roots/phm2012
```

这两个路径一般是软链接，指向仓库外部真实数据目录。可以用下面命令确认：

```bash
readlink data/loader_roots/xjtu
readlink data/loader_roots/phm2012/Learning_set
readlink data/loader_roots/phm2012/Full_Test_Set
```

## 3. XJTU-SY 当前划分

配置文件：

```text
conf/split/xjtu_bearing_index_split.yaml
```

当前配置：

```yaml
name: xjtu_bearing_index_split
enabled: true
condition_ids: []
train_bearing_indices: [1, 2, 3]
val_bearing_indices: [4]
test_bearing_indices: [5]
```

`condition_ids: []` 表示不限制单一工况，而是在所有 XJTU-SY 工况上按 bearing 后缀统一划分：

| Split | 轴承 |
|---|---|
| train | `Bearing1_1`, `Bearing1_2`, `Bearing1_3`, `Bearing2_1`, `Bearing2_2`, `Bearing2_3`, `Bearing3_1`, `Bearing3_2`, `Bearing3_3` |
| val | `Bearing1_4`, `Bearing2_4`, `Bearing3_4` |
| test | `Bearing1_5`, `Bearing2_5`, `Bearing3_5` |

这对应用户指定的逻辑：跨工况后，后缀 `1/2/3` 训练，后缀 `4` 验证，后缀 `5` 测试。

### 运行命令

```bash
uv run python -m USTC.SSE.BearingPrediction.cli.main \
  mode=build_index \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  dataset.root=/Users/nev8r/Desktop/phm2/data/loader_roots/xjtu \
  project.artifact_root=artifacts/load_demo \
  run.name=load_xjtu_bearing_index
```

### 当前本地加载结果

| 项目 | 结果 |
|---|---:|
| index rows | 9216 |
| train samples | 7032 |
| val samples | 1679 |
| test samples | 505 |
| raw sample shape | `(32768, 2)` |
| raw channels | `["h", "v"]` |

按轴承展开：

| Split | 工况 | 轴承 | sample 数 |
|---|---|---|---:|
| train | 35Hz12kN | Bearing1_1 | 123 |
| train | 35Hz12kN | Bearing1_2 | 161 |
| train | 35Hz12kN | Bearing1_3 | 158 |
| train | 37.5Hz11kN | Bearing2_1 | 491 |
| train | 37.5Hz11kN | Bearing2_2 | 161 |
| train | 37.5Hz11kN | Bearing2_3 | 533 |
| train | 40Hz10kN | Bearing3_1 | 2538 |
| train | 40Hz10kN | Bearing3_2 | 2496 |
| train | 40Hz10kN | Bearing3_3 | 371 |
| val | 35Hz12kN | Bearing1_4 | 122 |
| val | 37.5Hz11kN | Bearing2_4 | 42 |
| val | 40Hz10kN | Bearing3_4 | 1515 |
| test | 35Hz12kN | Bearing1_5 | 52 |
| test | 37.5Hz11kN | Bearing2_5 | 339 |
| test | 40Hz10kN | Bearing3_5 | 114 |

## 4. PHM2012 当前划分

配置文件：

```text
conf/split/phm2012_official.yaml
```

当前配置使用显式 bearing 列表：

```yaml
name: phm2012_official
enabled: true
mode: explicit
```

划分规则：

| Split | 轴承 |
|---|---|
| train | `Bearing1_1`, `Bearing1_2`, `Bearing2_1`, `Bearing2_2`, `Bearing3_1`, `Bearing3_2` |
| val | `Bearing1_3`, `Bearing2_3` |
| test | `Bearing1_4`, `Bearing1_5`, `Bearing1_6`, `Bearing1_7`, `Bearing2_4`, `Bearing2_5`, `Bearing2_6`, `Bearing2_7`, `Bearing3_3` |

解释：

- train 使用官方 `Learning_set` 中的 6 个轴承。
- val 从官方 `Full_Test_Set` 中抽出 `Bearing1_3`、`Bearing2_3`。
- test 使用剩余 `Full_Test_Set` 轴承。
- 这样不会从训练轴承中再切验证集，保留完整 Learning_set 用于训练侧。

### 运行命令

```bash
uv run python -m USTC.SSE.BearingPrediction.cli.main \
  mode=build_index \
  dataset=phm2012 \
  split=phm2012_official \
  dataset.root=/Users/nev8r/Desktop/phm2/data/loader_roots/phm2012 \
  project.artifact_root=artifacts/load_demo \
  run.name=load_phm2012_official
```

### 当前本地加载结果

| 项目 | 结果 |
|---|---:|
| index rows | 24889 |
| train samples | 7534 |
| val samples | 4330 |
| test samples | 13025 |
| raw sample shape | `(2560, 2)` |
| raw channels | `["h", "v"]` |

按轴承展开：

| Split | source_group | 工况 | 轴承 | sample 数 |
|---|---|---|---|---:|
| train | Learning_set | Condition1 | Bearing1_1 | 2803 |
| train | Learning_set | Condition1 | Bearing1_2 | 871 |
| train | Learning_set | Condition2 | Bearing2_1 | 911 |
| train | Learning_set | Condition2 | Bearing2_2 | 797 |
| train | Learning_set | Condition3 | Bearing3_1 | 515 |
| train | Learning_set | Condition3 | Bearing3_2 | 1637 |
| val | Full_Test_Set | Condition1 | Bearing1_3 | 2375 |
| val | Full_Test_Set | Condition2 | Bearing2_3 | 1955 |
| test | Full_Test_Set | Condition1 | Bearing1_4 | 1428 |
| test | Full_Test_Set | Condition1 | Bearing1_5 | 2463 |
| test | Full_Test_Set | Condition1 | Bearing1_6 | 2448 |
| test | Full_Test_Set | Condition1 | Bearing1_7 | 2259 |
| test | Full_Test_Set | Condition2 | Bearing2_4 | 751 |
| test | Full_Test_Set | Condition2 | Bearing2_5 | 2311 |
| test | Full_Test_Set | Condition2 | Bearing2_6 | 701 |
| test | Full_Test_Set | Condition2 | Bearing2_7 | 230 |
| test | Full_Test_Set | Condition3 | Bearing3_3 | 434 |

## 5. 查看生成产物

运行 `mode=build_index` 后，每次会在 `project.artifact_root/runs/<run_id>/` 下生成：

```text
index/sample_index.parquet
index/sample_index.csv
index/index_report.json
split/split.json
split/split_report.json
config/resolved.yaml
run.json
validation_report.json
```

最重要的是：

| 文件 | 用途 |
|---|---|
| `index/sample_index.parquet` | 每个原始 CSV sample 的完整索引 |
| `split/split.json` | train/val/test 的 `sample_uid` 列表和 bearing 列表 |
| `split/split_report.json` | 检查是否有 sample overlap、bearing overlap、空 split |

## 6. 程序化加载示例

如果不走 CLI，也可以直接调用当前基础设施：

```python
from pathlib import Path
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
from USTC.SSE.BearingPrediction.infra.split.SplitRegistry import build_splitter
from USTC.SSE.BearingPrediction.infra.feature.RawSampleReader import RawSampleReader

root = Path("/Users/nev8r/Desktop/phm2")
cfg = OmegaConf.create({
    "dataset": OmegaConf.load(root / "conf/dataset/xjtu_sy.yaml"),
    "split": OmegaConf.load(root / "conf/split/xjtu_bearing_index_split.yaml"),
})
cfg.dataset.root = str(root / "data/loader_roots/xjtu")

index = IndexBuilder().build(cfg)
split = build_splitter(cfg.split).split(index)

sample_row = index[index["sample_uid"] == split.train_sample_uids[0]].iloc[0]
signal, channels = RawSampleReader().read(sample_row)
```

PHM2012 只需要把配置替换为：

```python
cfg = OmegaConf.create({
    "dataset": OmegaConf.load(root / "conf/dataset/phm2012.yaml"),
    "split": OmegaConf.load(root / "conf/split/phm2012_official.yaml"),
})
cfg.dataset.root = str(root / "data/loader_roots/phm2012")
```

## 7. 与 `data/train` 软链接目录的关系

`data/train` 下存在历史/辅助软链接划分目录，适合人工浏览或旧 loader 实验。但当前 Stage 0+ 主线推荐使用本文件中的 index 级划分：

```text
dataset.root + split config
```

原因是 index 级划分可以把同一份 `SplitResult` 传给特征、标签、任务和训练流程，减少目录软链接和实验配置不一致的问题。
