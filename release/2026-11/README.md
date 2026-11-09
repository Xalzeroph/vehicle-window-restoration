# TintedGlass / Filmed-Glass Reflection Removal — Research Release (2026-11)

Code, data notes, and paper drafts from a study of **single-image reflection removal (SIRR) through filmed / tinted automotive glass**, captured with a controlled black-cloth protocol (paired *input* and *ground-truth transmission* frames).

Release branch: **`release/2026-11`** of `Xalzeroph/vehicle-window-restoration`.
Maintainer contact: 周明彧 (mingyuzhou541@gmail.com).

> **Scope and honesty note.** This release contains *no* new state-of-the-art method. It releases (i) the capture/QC protocol, (ii) the diagnostic benchmark and analysis code, and (iii) two paper drafts whose negative and boundary results are reported in full. Numbers marked as retracted in the paper drafts must not be cited (see `paper/cvpr2027_retracted_draft/README_RETRACTED.md`).

## Contents

| Path | What it is |
|---|---|
| `code/` | Reproducible research code (dataset builders, training, inference, evaluation, Ω mask utilities, our RDNet fork). Paths are parameterized via `TINTED_ROOT`, `TINTED_THIRD_PARTY`, `TINTED_PYTHON` (see `code/env.example.sh`). |
| `docs/` | Experiment ledger, result summaries, QC runbook, capture SOP, benchmark plan, novelty/venue notes, and the pre-registration documents that fix every decision rule *before* running. |
| `paper/icme2027/` | **ICME 2027 working draft** (“Plan B”: benchmark + diagnostic study), 6 pages incl. references + 4-page supplement; frozen 2026-11-06. |
| `paper/cvpr2027_retracted_draft/` | Historical CVPR draft (Oct 2026). **Contains retracted claims** (see `README_RETRACTED.md`); archived for provenance only. |
| `DATA.md` | Dataset card / layout for the November 2026 batch, plus what is and is not ground truth. |
| `RELEASE_NOTES.md` | What was included, what was cleaned, known issues. |

## Data availability

The November 2026 capture batch (**15 capture sessions of one filmed-glass pane**) is distributed separately:

- Hugging Face dataset: `zmy1234567890/filmed-glass-nov02` (private during pre-release review; request access).
- `I/` input frames, `gt/` 15 black-cloth ground-truth frames, `meta.json` per-pair metadata.
- `R/` and `omega/` are **auxiliary pseudo-labels**, not ground truth (see `DATA.md`).
- The batch has **no glass-angle metadata**; angle-related analyses are therefore not claimed (documented in `docs/nov02_report_v3.md`).

Other benchmarks (Real20, SIR², OpenRR-5k, RRW) are **not** redistributed; download them from their original sources.

## Reproduce (high level)

```bash
# 0) environment: Python 3.12, torch 2.12+cu126, numpy, opencv-python, PyMuPDF
export TINTED_ROOT=/path/to/project            # git-style project root
export TINTED_THIRD_PARTY=/path/to/third_party # RDNet weights / baselines / datasets
PY=/path/to/python

# 1) evaluation table on the 15-scene panel (needs predictions or checkpoints)
$PY code/eval/table_final_1019.py

# 2) single-arm inference example
$PY code/infer/eval_arm_1019.py --ckpt <ckpt.pth> --name <tag>

# 3) build the ICME draft (uses the bundled IEEEtran/IEEEbib in paper/icme2027/texinputs)
cd paper/icme2027 && bash -lc 'pdflatex main && bibtex main && pdflatex main && pdflatex main'
```

The papers are built with pdfLaTeX; the ICME draft ships the official IEEEtran v1.8b class and IEEEbib.bst from the ICME 2026 author kit.

## What the study found (short version)

1. **A reproducible filmed-glass paired benchmark** (black-cloth GT, per-pair QC, scene-level splits) — small (15 scenes) but fully documented.
2. **Two failure axes of SIRR on filmed glass:** (a) the global photometric level of the transmission layer is not identifiable from a single frame; (b) inside the occluder region the *local content* is partly unrecoverable (reflection dominance), and all evaluated models perform worse there than the identity baseline.
3. **A unified-recipe control experiment** (20 trained arms × 3 seeds) showing that the earlier “+0.50 dB” benefit of an occlusion-aware regularizer does **not** survive fair comparison: with one recipe, full − base = +0.19 dB, full − identity-regularizer = +0.079 dB, and full is significantly *worse* than a hard-threshold control. Run-level noise alone is ~0.9 dB on this setup.
4. **Boundary results**: LightRR (9.5 M, 102.5k fine-tuning iterations) reaches 20.22 dB on our panel versus 22.2–22.6 dB for the strong RDNet baseline; a small student distilled from RDNet and a coarse-to-fine student did not close the gap under matched-compute controls.

No public-benchmark SOTA claim is made anywhere in this release.

## Integrity and provenance

- Every decision rule (acceptance thresholds, statistics, stopping rules) lives in `docs/prereg/` and was written **before** the corresponding run; the experiment ledger records outcomes and retractions in place.
- Third-party user names, absolute workstation paths, and server-specific details were redacted before publication.
- Model weights are **not** included: RDNet/LightRR official weights and third-party baselines remain under their original licenses and distribution terms.
- `code/RDNet/` is a fork of the upstream RDNet code base; see that directory and `code/MANIFEST.md` for the change list. The IEEEtran class files in `paper/icme2027/texinputs/` come from the ICME 2026 author kit and remain under the IEEE distribution terms.

## License / usage

Research use only. No license is granted for third-party components beyond their own terms. If you use the benchmark protocol or code, please cite the draft (see each paper’s README once a venue is final). Do not cite retracted numbers.

## Citation (placeholder)

```bibtex
@misc{tintedglass2026release,
  author = {Zhou, Mingyu and contributors},
  title  = {TintedGlass: single-image reflection removal through filmed glass --- research release},
  year   = {2026},
  note   = {Code, protocol and drafts. Branch release/2026-11}
}
```
