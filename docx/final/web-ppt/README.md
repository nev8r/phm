# 结题答辩网页 PPT

推荐打开：

```bash
open docx/final/web-ppt/index.html
```

该版本使用 `op7418/guizang-ppt-skill` 的 Swiss / IKB 模板生成，适合作为最后汇报主版本。新版共 14 页，重点展示真实数据特征曲线、系统实现、论文复现实验和测试验收证据。传统 `.pptx` 文件仍保留在 `docx/final/工业轴承设备剩余寿命预测系统的实现-结题答辩.pptx`，用于需要 Office 文件时备用。

重新生成：

```bash
uv run python scripts/generate_final_web_ppt.py
node .agents/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs docx/final/web-ppt/index.html
```

重新生成时，脚本会从 `data/external` 本地真实数据生成以下证据图：

- `images/05-xjtu-bearing1-1-rms-health.png`
- `images/06-phm2012-bearing1-1-rms-health.png`
