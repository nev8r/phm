# 轴承寿命预测与故障诊断系统：编码规范文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v1.0 |
| 日期 | 2026年6月 |


## 目录与命名规范

源码放在 `src/USTC/SSE/BearingPrediction`，示例中通过 `phm` 导入。模块按职责分为 `data`、`model`、`engine`、`util`，测试放在 `tests`，Notebook 放在 `examples`，课程文档放在 `docx`。

## 导入规范

Notebook、用户手册和演示代码推荐使用：

```python
from phm.data.loader.PHM2012Loader import PHM2012Loader
from phm.model.basic.CNN import CNN
```

内部历史路径 `USTC.SSE.BearingPrediction` 作为物理包路径保留，不在示例中作为首选导入方式。

## 文件头规范

所有 `src` 与 `tests` 下 Python 文件使用统一模块 docstring，包含模块用途、文件说明、创建者、版权和年份。作者分配为：`data/**` 由 cyj，`engine/**` 由 zdh，`model/**` 和包入口由 zyj，`util/**` 与 `tests/**` 由 zy。

## 代码风格

- Python 版本固定为 3.11。
- 包管理使用 uv，不再混用 pipenv、poetry 或 requirements 手工安装流程。
- 路径使用 `pathlib.Path`，不得硬编码个人磁盘路径。
- 设备选择通过工具函数或显式配置处理，Notebook 中需说明 CPU/MPS/CUDA 的差异。
- 大数据、缓存、训练产物不进入源码包；文档只引用必要指标和图表。
