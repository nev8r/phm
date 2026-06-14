# 结题答辩网页 PPT

推荐打开：

```bash
open docx/final/web-ppt/index.html
```

该版本使用 `op7418/guizang-ppt-skill` 的 Swiss / IKB 模板生成，适合作为最后汇报主版本。传统 `.pptx` 文件仍保留在 `docx/final/工业轴承设备剩余寿命预测系统的实现-结题答辩.pptx`，用于需要 Office 文件时备用。

重新生成：

```bash
uv run python scripts/generate_final_web_ppt.py
node .agents/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs docx/final/web-ppt/index.html
```
