# 轴承寿命预测与故障诊断系统：安装配置手册

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v1.0 |
| 日期 | 2026年6月 |


## 环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | macOS 或常见 Linux/Windows 开发环境 |
| Python | 3.11 |
| 包管理 | uv |
| 深度学习 | PyTorch 2.10 依赖范围，Mac 可使用 MPS |
| 交互环境 | Jupyter / ipykernel，可选 |

## 安装步骤

```bash
cd /Users/nev8r/Desktop/main
uv sync
uv run phm --help
uv run python -m unittest discover -v
```

## 数据配置

数据不放入 Git 仓库。当前推荐路径如下：

| 任务 | 路径 |
|---|---|
| PHM2012 | `data/loader_roots/phm2012` |
| XJTU-SY | `data/loader_roots/xjtu` |

如需重新映射，使用软链接指向本机外部数据目录，并保持 CLI、示例和测试脚本都访问 `data/loader_roots`。

## 常见问题

| 问题 | 处理 |
|---|---|
| 找不到数据 | 检查软链接是否存在、目标目录是否可读 |
| 找不到 sklearn | 执行 `uv sync`，确认 `scikit-learn` 在依赖中 |
| TensorBoard 不可用 | 确认 `tensorboard` 已由 uv 安装 |
| MPS 不可用 | 退回 CPU 或在配置中显式选择可用设备 |
