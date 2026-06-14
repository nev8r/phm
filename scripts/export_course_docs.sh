#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --pdf-engine=xelatex
  -V "CJKmainfont=Songti SC"
  -V "mainfont=Times New Roman"
  -V "monofont=Menlo"
  -V "geometry:top=2.54cm,bottom=2.54cm,left=2.54cm,right=2.54cm"
  -V "colorlinks=false"
  -V "papersize:a4"
  -V "fontsize=10.5pt"
  -V "toc-title=目录"
  -H "scripts/ustc_course_pdf_style.tex"
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
    markdown_dir="$(dirname "$markdown_file")"
    resource_path=".:${markdown_dir}"
    pandoc "$markdown_file" --resource-path="$resource_path" -o "${output_dir}/${base_name}.pdf" "${COMMON_ARGS[@]}"
    pandoc "$markdown_file" --resource-path="$resource_path" -o "${output_dir}/${base_name}.docx" "${DOCX_ARGS[@]}"
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
