# IEEE report package

This folder contains the report in the standard IEEE conference layout.

## Files

- `main.tex` - IEEEtran conference source
- `references.bib` - BibTeX bibliography
- `../output/pdf/A_Configurable_Hybrid_Guardrail_Architecture.pdf` - final compiled report

## Local build
Run the following command from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=A_Configurable_Hybrid_Guardrail_Architecture \
  -outdir=../output/pdf main.tex
```

## Overleaf

1. Create a new blank project in Overleaf.
2. Upload `main.tex` and `references.bib`.
3. Set `main.tex` as the main document.
4. Select **pdfLaTeX** as the compiler.
5. Use **Recompile from scratch** if the bibliography is unresolved on the first run.

The document uses `\documentclass[conference]{IEEEtran}` and the IEEE BibTeX style.

## Evidence note

The report uses the calibration evidence documented in the repository and
states that the frozen 400-case holdout remains unopened.
