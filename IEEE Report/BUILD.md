# Building the paper

- `access.tex`: the paper (IEEE Access template, filled). Original template: `access_template.tex` / `access_template.pdf`.
- `figures/`: all figures, vector PDF (white background, embedded Arial-style fonts) + 600-dpi PNG.
  Regenerate them from the project outputs with `python scripts/make_paper_figures.py`.
- `paper_numbers.json`: every number quoted in the paper, computed from `reports/evaluation/episodes.csv`
  and `reports/weather_validation/summary.json`.

## Compile (pdfLaTeX — required by ieeeaccess.cls)

Overleaf: upload this folder, set the compiler to **pdfLaTeX**, compile `access.tex`.
Local (TeX Live / MiKTeX): run `pdflatex access.tex` twice.

`access_preview.pdf` is a preview built with XeTeX (Tectonic) using a copy of the class with the
Pantone spot colour replaced by CMYK, because XeTeX cannot use the `spotcolor` package. Bold and italic
Times may look slightly different from the final pdfLaTeX build. Use the pdfLaTeX output for submission.

## Still to fill in before submission
- Author names, affiliations, e-mails, corresponding author (top of `access.tex`, marked TODO).
- Author biographies and photos (end of `access.tex`, marked TODO).
- Funding footnote (`\tfootnote`) and the acknowledgment, if needed.
