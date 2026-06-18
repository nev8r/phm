# Data Train Splits

更新日期：2026-06-19

本目录固定项目后续训练/验证/测试使用的数据划分。目录下只放软链接，不复制原始数据。

原始入口仍在：

- `data/loader_roots/phm2012`
- `data/loader_roots/xjtu`

## 1. 总体原则

| 数据集 | 固定划分方式 | 说明 |
|---|---|---|
| PHM2012 / PRONOSTIA | 官方 train/test | `Learning_set` 作为训练侧，`Full_Test_Set` 作为测试侧 |
| XJTU-SY | 同工况 leave-one-bearing-out | 每个工况 5 个轴承，每次留 1 个测试，剩余 4 个中 1 个验证、3 个训练 |

## 2. PHM2012 划分

路径：

```text
data/train/phm2012/official/
├── train/
│   ├── Learning_set/
│   └── Full_Test_Set/
└── test/
    ├── Learning_set/
    └── Full_Test_Set/
```

固定规则：

- `train/Learning_set` 链接官方 `Learning_set` 中 6 个轴承。
- `test/Full_Test_Set` 链接官方 `Full_Test_Set` 中 11 个轴承。
- 不额外固定验证集，避免偏离官方 train/test 划分；若训练流程需要 validation，应只从 `train/Learning_set` 内部再切分。

当前 PHM2012 训练轴承：

```text
Bearing1_1
Bearing1_2
Bearing2_1
Bearing2_2
Bearing3_1
Bearing3_2
```

当前 PHM2012 测试轴承：

```text
Bearing1_3
Bearing1_4
Bearing1_5
Bearing1_6
Bearing1_7
Bearing2_3
Bearing2_4
Bearing2_5
Bearing2_6
Bearing2_7
Bearing3_3
```

## 3. XJTU-SY 划分

路径：

```text
data/train/xjtu/leave_one_bearing_out/<condition>/fold_test_<bearing>/
├── train/
├── val/
└── test/
```

固定规则：

- 每个工况单独做 5 折。
- 每折测试集为当前 fold 名中的轴承。
- 验证集为同一工况下按编号顺序的下一个轴承，循环选择。
- 训练集为同一工况下剩余 3 个轴承。
- 每个 `train`、`val`、`test` 目录都保留 `35Hz12kN`、`37.5Hz11kN`、`40Hz10kN` 工况目录形状；实际软链接只放在该 fold 对应工况下。

### 35Hz12kN

| Fold | Train | Val | Test |
|---|---|---|---|
| `fold_test_Bearing1_1` | Bearing1_3, Bearing1_4, Bearing1_5 | Bearing1_2 | Bearing1_1 |
| `fold_test_Bearing1_2` | Bearing1_1, Bearing1_4, Bearing1_5 | Bearing1_3 | Bearing1_2 |
| `fold_test_Bearing1_3` | Bearing1_1, Bearing1_2, Bearing1_5 | Bearing1_4 | Bearing1_3 |
| `fold_test_Bearing1_4` | Bearing1_1, Bearing1_2, Bearing1_3 | Bearing1_5 | Bearing1_4 |
| `fold_test_Bearing1_5` | Bearing1_2, Bearing1_3, Bearing1_4 | Bearing1_1 | Bearing1_5 |

### 37.5Hz11kN

| Fold | Train | Val | Test |
|---|---|---|---|
| `fold_test_Bearing2_1` | Bearing2_3, Bearing2_4, Bearing2_5 | Bearing2_2 | Bearing2_1 |
| `fold_test_Bearing2_2` | Bearing2_1, Bearing2_4, Bearing2_5 | Bearing2_3 | Bearing2_2 |
| `fold_test_Bearing2_3` | Bearing2_1, Bearing2_2, Bearing2_5 | Bearing2_4 | Bearing2_3 |
| `fold_test_Bearing2_4` | Bearing2_1, Bearing2_2, Bearing2_3 | Bearing2_5 | Bearing2_4 |
| `fold_test_Bearing2_5` | Bearing2_2, Bearing2_3, Bearing2_4 | Bearing2_1 | Bearing2_5 |

### 40Hz10kN

| Fold | Train | Val | Test |
|---|---|---|---|
| `fold_test_Bearing3_1` | Bearing3_3, Bearing3_4, Bearing3_5 | Bearing3_2 | Bearing3_1 |
| `fold_test_Bearing3_2` | Bearing3_1, Bearing3_4, Bearing3_5 | Bearing3_3 | Bearing3_2 |
| `fold_test_Bearing3_3` | Bearing3_1, Bearing3_2, Bearing3_5 | Bearing3_4 | Bearing3_3 |
| `fold_test_Bearing3_4` | Bearing3_1, Bearing3_2, Bearing3_3 | Bearing3_5 | Bearing3_4 |
| `fold_test_Bearing3_5` | Bearing3_2, Bearing3_3, Bearing3_4 | Bearing3_1 | Bearing3_5 |

## 4. 使用示例

PHM2012 官方训练侧：

```python
from phm.data.loader.PHM2012Loader import PHM2012Loader

loader = PHM2012Loader("data/train/phm2012/official/train")
```

PHM2012 官方测试侧：

```python
from phm.data.loader.PHM2012Loader import PHM2012Loader

loader = PHM2012Loader("data/train/phm2012/official/test")
```

XJTU-SY 某个 fold 的训练/验证/测试：

```python
from phm.data.loader.XJTULoader import XJTULoader

base = "data/train/xjtu/leave_one_bearing_out/35Hz12kN/fold_test_Bearing1_1"
train_loader = XJTULoader(f"{base}/train")
val_loader = XJTULoader(f"{base}/val")
test_loader = XJTULoader(f"{base}/test")
```

## 5. 注意事项

- 本目录下的轴承目录都是软链接，删除这些链接不会删除原始数据。
- PHM2012 的验证集不在本目录固定，训练时如果需要验证集，只能从官方训练侧内部切分。
- XJTU-SY 的 fold 只在同一工况内划分，不跨工况混合。
- 后续实验、Notebook 和训练配置应优先引用本目录，而不是直接手写原始 `data/loader_roots` 路径。
