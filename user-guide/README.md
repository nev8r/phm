# User Guide

更新日期：2026-06-19

本目录集中放项目用户侧文档。数据集说明、加载方式、实验划分分别放在不同文件里，避免把“数据本身是什么”和“项目怎么加载/切分”混在一起。

## 文档索引

| 文档 | 内容 |
|---|---|
| [xjtu-sy-dataset-card.md](xjtu-sy-dataset-card.md) | XJTU-SY 数据集 card：官方采集信息、本地目录、轴承清单、当前框架读取后的样子 |
| [phm2012-dataset-card.md](phm2012-dataset-card.md) | PHM2012 / PRONOSTIA 数据集 card：官方采集信息、本地目录、轴承清单、当前框架读取后的样子 |
| [loading-and-splits.md](loading-and-splits.md) | 当前项目推荐的加载入口、CLI 命令、XJTU-SY 与 PHM2012 的固定划分结果 |

## 当前推荐入口

当前 Stage 0+ 框架的主线入口不是手动遍历 `data/train` 软链接目录，而是：

```text
dataset.root
→ IndexBuilder
→ sample_index
→ SplitRegistry / SplitResult
→ FeatureExtractor / LabelBuilder / TaskBuilder
```

其中 `sample_index` 记录每个原始 CSV 快照的 `sample_uid`、`bearing_id`、`condition_id`、`file_path` 等字段；`SplitResult` 用 `sample_uid` 列表定义 train/val/test。后续特征、标签、窗口任务都会继承这份划分。

## 本地数据入口

```text
data/loader_roots/xjtu
data/loader_roots/phm2012
```

这两个路径通常是软链接，指向仓库外部的真实大数据目录。原始数据不应提交到 Git。
