# 结项交付清单

本文档依据 SRS、确认测试计划、课程阶段要求和结题测试报告汇总最终交付物。

| 课程要求 | 仓库交付物 | 状态 |
| --- | --- | --- |
| 源代码 | `src/USTC/SSE/BearingPrediction` | 已完成 |
| 开题报告 | `docx/proposal/01_开题报告.pdf` | 已完成 |
| 需求定义文档 | `docx/proposal/04_需求定义文档.pdf` | 已完成 |
| SRS | `docx/proposal/05_SRS规格说明文档.pdf` | 已完成 |
| 确认测试计划 | `docx/proposal/09_确认测试计划文档.pdf` | 已完成 |
| 项目管理计划 | `docx/proposal/10_项目管理计划文档.pdf` | 已完成 |
| 中期检查报告 | `docx/mid-term/02_中期检查报告.pdf` | 已完成 |
| 设计文档 | `docx/mid-term/06_设计文档.pdf` | 已完成 |
| 单元测试计划 | `docx/mid-term/07_单元测试计划文档.pdf` | 已完成 |
| 集成测试计划 | `docx/mid-term/08_集成测试计划文档.pdf` | 已完成 |
| 编码规范 | `docx/mid-term/11_编码规范文档.pdf` | 已完成 |
| 结题报告 | `docx/final/12_结题报告.pdf` | 已完成 |
| 单元测试报告 | `docx/final/13_单元测试报告.pdf` | 已完成 |
| 集成测试报告 | `docx/final/14_集成测试报告.pdf` | 已完成 |
| 确认测试报告 | `docx/final/15_确认测试报告.pdf` | 已完成 |
| 用户使用手册 | `docx/final/16_用户使用手册.pdf` | 已完成 |
| 安装配置手册 | `docx/final/17_安装配置手册.pdf` | 已完成 |
| 项目技术论文 | `docx/final/18_项目技术论文.pdf` | 已完成 |
| 成员贡献比说明 | `docx/final/19_成员贡献比说明.pdf` | 已完成 |
| 结题答辩提纲 | `docx/final/20_结题答辩提纲.pdf` | 已完成 |
| 结题答辩演讲稿 | `docx/final/21_结题答辩演讲稿.pdf` | 已完成 |
| 系统从 0 到 1 总览说明 | `docx/final/22_系统从0到1总览说明.pdf` | 已完成 |
| 数据语义与样本构造说明 | `docx/final/23_数据语义与样本构造说明.pdf` | 已完成 |
| 端到端运行与输出解读手册 | `docx/final/24_端到端运行与输出解读手册.pdf` | 已完成 |
| 模型与实验设计说明 | `docx/final/25_模型与实验设计说明.pdf` | 已完成 |
| 验证证据与追踪矩阵 | `docx/final/26_验证证据与追踪矩阵.pdf` | 已完成 |
| 生存分析与失效概率范围说明 | `docx/final/27_生存分析与失效概率范围说明.pdf` | 已完成 |
| 指标驱动实验结果说明与 Open-Source SOTA 对照 | `docs/project-owner/08_指标驱动实验结果说明.md`、`docs/reproduction-evidence/open_source_sota_*.csv` | SOTA target 与 gap 证据已建立；RULSurv RSF port 已完成，AutoRUL/GNN/Weibull 仍需独立环境 |
| 结题答辩 PPT（推荐） | `docx/final/web-ppt/index.html` | 已完成 |
| 结题答辩 PPT（备用） | `docx/final/工业轴承设备剩余寿命预测系统的实现-结题答辩.pptx` | 已完成 |
| 用户示例 | `examples/*.ipynb` | 已完成 |
| 测试报告依据 | `tests`、pytest 输出、`docs/PAPER_REPRODUCTION.md` | 已完成 |
| 论文复现说明 | `docs/PAPER_REPRODUCTION.md` | 已完成 |
| 真实训练证据摘要 | `docs/reproduction-evidence/*.csv`、`docs/reproduction-evidence/README.md` | 已完成论文复现摘要；新增 RULSurv RSF port 与 SOTA 对照摘要 |
| 项目 owner 工程阅读版 | `docs/project-owner/*.md` | 已完成 |

## 验收命令

```bash
uv run --extra dev pytest tests/test_rul_metrics.py tests/test_paper_cnn_lstm_attention.py tests/test_paper_xlstm_transformer.py tests/test_examples_notebooks.py -q
uv run --extra dev pytest -q
bash scripts/export_course_docs.sh
uv run python scripts/generate_final_web_ppt.py
test -f docx/final/web-ppt/index.html
uv run python scripts/generate_final_ppt.py
```

## 验收结果摘要

| 验收项 | 最近结果 |
| --- | --- |
| Focused 测试 | `20 passed in 4.26s` |
| notebook 测试 | `4 passed in 27.57s` |
| 全量测试 | `39 passed in 28.93s` |
| 网页 PPT 校验 | `Swiss deck validation passed: 16 slide(s)` |
| 文档导出 | `bash scripts/export_course_docs.sh` 完成，结题阶段 17 份 PDF + 17 份 DOCX |
| 论文复现证据 | 真实训练输出位于本机 `tmp/`，提交包保留 `docs/reproduction-evidence` 摘要 |
| Open-Source SOTA 对照 | target/gap 表已生成；RULSurv RSF port 原协议 row-level CV mean true MAE `10.244649` min，低于 target `12.6` min，状态 `PROTOCOL_PASS`；该结果不等价于 held-out-bearing 泛化，AutoRUL/GNN/Weibull 仍为后续硬门槛 |

## 归档建议

最终提交或压缩归档时应包含源码、测试、notebook、正式 `docs` 文档、`docx/proposal`、`docx/mid-term`、`docx/final` 和脚本；不包含 `tmp/`、`outputs/`、`data/external/`、`runs/`、`checkpoints/` 等运行产物。若本地存在内部工作计划目录，不纳入课程交付包。
