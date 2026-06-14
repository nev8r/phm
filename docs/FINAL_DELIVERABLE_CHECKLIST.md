# 结项交付清单

本文档依据 `AGENTS.md`、SRS 和 `docs/工程实践各阶段要求.pdf` 汇总最终交付物。

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
| 结题答辩 PPT | `docx/final/工业轴承设备剩余寿命预测系统的实现-结题答辩.pptx` | 已完成 |
| 用户示例 | `examples/*.ipynb` | 已完成 |
| 测试报告依据 | `tests`、pytest 输出、`docs/PAPER_REPRODUCTION.md` | 已完成 |
| 论文复现说明 | `docs/PAPER_REPRODUCTION.md` | 已完成 |

## 验收命令

```bash
uv run --extra dev pytest tests/test_rul_metrics.py tests/test_paper_cnn_lstm_attention.py tests/test_paper_xlstm_transformer.py tests/test_examples_notebooks.py -q
uv run --extra dev pytest -q
bash scripts/export_course_docs.sh
uv run python scripts/generate_final_ppt.py
```

## 归档建议

最终提交或压缩归档时应包含源码、测试、notebook、`docs`、`docx/proposal`、`docx/mid-term`、`docx/final` 和脚本；不包含 `tmp/`、`outputs/`、`data/external/`、`runs/`、`checkpoints/` 等运行产物。
