# ⚠️ RETRACTED DRAFT — archival copy, do not cite

This is the **CVPR 2027 draft frozen in October 2026** (title: *What Single-Image Reflection Removal Does to the Occluder Region: A Diagnostic Benchmark on Tinted Glass*). It is included only so that the project’s history is auditable. **Several of its headline numbers are known to be wrong and were formally retracted.**

## What was retracted

1. **“+0.50 dB” benefit of the occlusion-aware / mask-free regularizer** (with a reported confidence interval and 14/15 win count, “5 seeds”).
2. The associated attribution that the improvement came from the occluder region or from the mask-free formulation.

## Why

A unified-recipe, 3-seed, 20-arm control experiment (pre-registered in `docs/prereg/PREREG_quiet_ablation_1106.md` and `AMEND_prereg_1106.md`) showed:

- The original “+0.50” compared **different training recipes and different provenance** (the historical baseline used an older script), not just the regularizer. Re-running the historical configuration reproduced its score, but re-running the **same seed and same script** on another date drifted by **−0.88 dB (base) / −0.51 dB (full)** — i.e., run-level noise alone is larger than the claimed effect.
- Under one recipe: full − base = **+0.19 dB**, full − identity-regularizer = **+0.079 dB** (< the pre-registered +0.26 dB gate), and full is **significantly worse** than a hard-threshold control (−0.246 dB). The historical +0.50 is therefore not a valid estimate of the regularizer’s effect.

## What remains usable

- The **capture protocol, benchmark panel, QC procedure and diagnostic analyses** (photometric-level non-identifiability, occlusion-region content loss, identity-baseline comparisons) are still valid and are carried into the ICME draft.
- The **negative/boundary results** in this draft are valid as records of what was run, but should be re-derived from the ledger (`docs/experiment_ledger.md`) rather than copied from the PDF.

## Do not

- Do not cite this draft’s aggregate numbers, confidence intervals, or win counts.
- Do not submit this draft anywhere.
- Do not merge its method sections into a new submission without replacing the retracted numbers with the unified-recipe results.

## Build

pdfLaTeX with the bundled CVPR author kit (`cvpr.sty`, `ieeenat_fullname.bst`): 6 pages body + 1 page references. The draft is frozen as-is; known TODOs (placeholder references, single-scene teaser, missing per-domain table) were never fixed.

Replacement authoritative material: `docs/results_summary_2026-11.md`, `docs/experiment_ledger.md` (E31–E51), and the ICME working draft at `paper/icme2027/`.
