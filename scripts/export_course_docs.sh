#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --pdf-engine=xelatex
  -V "CJKmainfont=PingFang SC"
  -V "mainfont=Times New Roman"
  -V "monofont=Menlo"
  -V "geometry:margin=2.2cm"
  -V "colorlinks=true"
  -V "linkcolor=blue"
  -V "urlcolor=blue"
  -V "papersize:a4"
  -V "fontsize:11pt"
  -V "toc-title=目录"
  --toc
  --number-sections
)

for markdown_file in docx/proposal/md/*.md; do
  base_name="$(basename "${markdown_file%.md}")"
  pandoc "$markdown_file" -o "docx/proposal/${base_name}.pdf" "${COMMON_ARGS[@]}"
done

for markdown_file in docx/mid-term/md/*.md; do
  base_name="$(basename "${markdown_file%.md}")"
  pandoc "$markdown_file" -o "docx/mid-term/${base_name}.pdf" "${COMMON_ARGS[@]}"
done
