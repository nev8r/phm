# 中文结题报告

本目录是 Step AE 的中文 LaTeX 结题报告。报告只整理最终论文正文内容，不新增训练，不制作 PPT，不复制模型权重或预测明细。

## 文件结构

- `main.tex`：报告入口。
- `sections/`：摘要到结论的 13 个章节文件。
- `tables/`：任务、数据集、主结果、复现实验等表格。
- `figures/`：报告使用的图片副本，均来自已验收的资产清单。
- `references.bib`：参考文献和项目内部报告来源。
- `build_check.md`：编译和版式检查记录。

## 编译

```bash
cd reports/final_defense/report
make
```

输出文件为 `main.pdf`。

## 内容口径

- RUL 统一使用 `linear_rul_norm`。
- GRU 200ep 是本文方法实验主线结果。
- 两个 50ep 视频只用于训练过程演示。
- 论文复现实验单独成节，不作为本文三任务主结果。
- 后续 Step AF 制作 USTC 风格 Beamer PPT 时，最后应加入任务分配页，四名成员各占 25% 贡献；具体姓名由最终答辩信息填入。
