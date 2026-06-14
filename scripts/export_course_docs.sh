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
)

DOCX_ARGS=(
  --toc
  -V "toc-title=目录"
)

export_stage() {
  local source_glob="$1"
  local output_dir="$2"
  local generated_docx=()

  mkdir -p "$output_dir"
  for markdown_file in $source_glob; do
    [[ -f "$markdown_file" ]] || continue
    base_name="$(basename "${markdown_file%.md}")"
    pandoc "$markdown_file" -o "${output_dir}/${base_name}.pdf" "${COMMON_ARGS[@]}"
    pandoc "$markdown_file" -o "${output_dir}/${base_name}.docx" "${DOCX_ARGS[@]}"
    generated_docx+=("${output_dir}/${base_name}.docx")
  done

  if ((${#generated_docx[@]} > 0)); then
    uv run --extra dev python scripts/apply_course_docx_style.py "${generated_docx[@]}"
  fi
}

export_stage "docx/proposal/md/*.md" "docx/proposal"
export_stage "docx/mid-term/md/*.md" "docx/mid-term"

if compgen -G "docx/final/md/*.md" > /dev/null; then
  export_stage "docx/final/md/*.md" "docx/final"
fi
