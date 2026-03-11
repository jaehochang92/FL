# Copilot Instructions for FL Codebase

## Overview
This project is a LaTeX-based academic paper focused on personalized federated learning with nonparametric empirical Bayes. The workspace contains the main manuscript, custom class files, a preamble, and bibliography. There are no build scripts or code files; the workflow is centered on LaTeX compilation.

## Key Files
- `main.tex`: Primary manuscript. Contains mathematical derivations, proofs, and section structure.
- `NSF.cls`: Custom class file for formatting. Defines document style and layout.
- `preamble.tex`: Shared LaTeX macros, packages, and settings. Included via `\input{preamble}` in `main.tex`.
- `references.bib`: BibTeX bibliography for citations.
- `%OUTDIR%/`: Output directory for build artifacts (aux, bbl, log, pdf, etc.).

## Developer Workflow
- **Build**: Compile `main.tex` using `pdflatex` or `latexmk`. Output files are written to `%OUTDIR%/`.
  - Example: `latexmk -pdf main.tex`
- **Bibliography**: Use BibTeX for references. Ensure `references.bib` is up to date.
- **Formatting**: Custom class (`NSF.cls`) and preamble (`preamble.tex`) control style and macros. Update these for global changes.
- **Debugging**: Check `.log` and `.synctex.gz` files in `%OUTDIR%/` for errors and warnings.

## Project Conventions
- Mathematical notation and sectioning follow academic standards for statistics and machine learning.
- Citations use `\citep` and `\citet` (natbib package).
- All custom macros and settings are centralized in `preamble.tex`.
- Output directory is named `%OUTDIR%/` (may be customized in build scripts).

## Integration Points
- No external code dependencies; all logic is in LaTeX and BibTeX files.
- Custom class and preamble files are tightly integrated with the manuscript.

## Examples
- To add a new macro, edit `preamble.tex` and reference it in `main.tex`.
- To update formatting, modify `NSF.cls`.
- To add a citation, update `references.bib` and use `\citep{key}` in `main.tex`.

## Recommendations for AI Agents
- Focus edits on `main.tex`, `preamble.tex`, and `NSF.cls` for content, macros, and formatting.
- When adding new sections or mathematical content, follow the notation and structure in `main.tex`.
- For bibliography changes, ensure BibTeX keys are consistent and update `references.bib`.
- When troubleshooting build errors, review `.log` files in `%OUTDIR%/`.
- When in doubt about formatting or conventions, refer to the existing content in `main.tex` and the settings in `NSF.cls` and `preamble.tex` for guidance.
- Do not remove any citations or references without confirming with the user, as they may be critical to the paper's content and argumentation.
- When answering in math, render your LaTeX codes to ensure readibility.

---

If any conventions or workflows are unclear, please request clarification or examples from the user.
