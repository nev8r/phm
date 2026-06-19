# LaTeX Feature Analysis Technical Report

This directory contains the LaTeX source for the completed bearing feature-analysis cycle.

## Scope

The report summarizes only archived results under:

```text
reports/feature_analysis/
```

It does not reference local artifact run directories, does not add new experiments, and does not include model-training results.

## Build

Recommended command:

```bash
make
```

Equivalent direct command:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Clean generated files:

```bash
make clean
```

The source uses `ctexart`, so compile with XeLaTeX.

## Contents

- `main.tex`: report entry point.
- `sections/`: one source file per chapter.
- `tables/`: compact report tables.
- `figures/`: selected copied figures from archived report directories.
- `references.bib`: project-internal references for archived report materials.
