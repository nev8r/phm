# 轴承寿命预测与故障诊断系统：单元测试计划文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v1.0 |
| 日期 | 2026年6月 |


## 测试原则

单元测试聚焦单个模块的输入输出、边界条件和维度约束，避免依赖完整大数据训练。测试命令为 `uv run python -m unittest discover -v`。

## 测试项

| 编号 | 模块 | 测试重点 |
|---|---|---|
| UT-01 | 特征处理器 | FFT、RMS、峭度、谱特征、频带能量输出维度与数值范围 |
| UT-02 | 数据集构造 | PHM2012/XJTU 样本 shape、标签 shape、元数据字段 |
| UT-03 | 模型 forward | 主线模型输出维度 |
| UT-04 | Trainer | 单轮训练、loss 记录、设备迁移 |
| UT-05 | 训练产物 | history、metrics、prediction 文件字段 |

## 通过标准

所有测试无 error、无 failure；模型 forward 不出现 shape mismatch；特征处理不产生非预期 NaN/Inf；测试不依赖个人机器绝对路径。
