# Release notes — 2026-11

## 2026-11-09 — initial public release (branch `release/2026-11`)

Included:

- `code/` — 172 files from the server project tree (data build, training, inference, evaluation, Ω utilities, RDNet fork).
- `docs/` — experiment ledger, results summary, QC runbook v1.2, capture SOP v1.1, benchmark plan v1.1, ICME 2027 rules, LightRR judge, nov02 report v3, data-side TODO, and 10 pre-registration documents.
- `paper/icme2027/` — clean build of the ICME 2027 working draft (6 pages incl. references; 4-page supplement), verified from a clean directory: 0 errors, 0 overfull boxes, 0 undefined citations.
- `paper/cvpr2027_retracted_draft/` — historical CVPR draft with a prominent retraction notice.
- `DATA.md` — dataset layout and ground-truth/pseudo-label semantics.

Cleaning performed before publication:

- Removed workstation-absolute paths from docs and scripts (scripts now use `TINTED_ROOT` / `TINTED_THIRD_PARTY` / `TINTED_PYTHON` with server defaults).
- Redacted a third-party server user name from a pre-registration document.
- Removed an internal build-helper directory and internal-only README/MANIFEST files.
- Excluded the figure that displayed retracted confidence intervals from the ICME package (only figures referenced by the frozen draft are shipped).

Known issues / caveats:

1. **Do not cite the retracted draft.** Its +0.50 dB claim was withdrawn after a unified-recipe, 3-seed control experiment; see `paper/cvpr2027_retracted_draft/README_RETRACTED.md`.
2. The ICME draft is a **working draft, not yet submitted**; it is publicly visible here and may be revised before submission. Check current venue rules before treating it as a preprint.
3. `paper/cvpr2027_retracted_draft/refs.bib` still contains placeholder entries for a few baselines (`Anonymous`/`arXiv`); it was frozen in that state and is archived as-is.
4. The CVPR draft has no supplementary material; `paper/icme2027/` does.
5. No model weights are included. No dataset files are included in git (the November batch is on Hugging Face; other benchmarks from their original sources).
6. `paper/icme2027/main.tex` keeps a harmless `\graphicspath` fallback to `../cvpr2027/fig/`; the clean package builds without it.
7. Repository `.gitignore` globally ignores `*.pdf`, `*.png`, `*.jpg`; image assets required for LaTeX builds were force-added. The rendered PDFs are therefore not tracked in git.
8. The November batch metadata lacks glass-angle labels, so all angle-related statements in docs are explicitly non-claims.
9. Real20 / SIR² / OpenRR / RRW subsets are not redistributed. Their licenses require separate download.

Data:

- `filmed-glass-nov02` (Hugging Face, private pending privacy review): 1499 input frames, 15 black-cloth GT frames, per-pair metadata; `R/` and `omega/` are auxiliary pseudo-labels and must not be described as ground truth.

Prior art / acknowledgements: the release builds on RDNet (CVPR 2025) as the strong baseline; the LightRR (CVPR 2026) code was used for the fine-tuning baseline and is not redistributed here.
