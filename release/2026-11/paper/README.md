# Paper drafts

Two LaTeX drafts are archived here. They are **research drafts**, not published papers.

## `icme2027/` — working draft (Plan B)

- Title: *What Filmed Glass Does to Single-Image Reflection Removal: A Development Benchmark and a Diagnostic Study*
- Status: **frozen working draft, 2026-11-06**, not yet submitted (ICME 2027 deadline: 2026-12-30 AoE).
- Length: 6 pages including references (ICME limit) + 4-page supplement.
- Claim structure: (1) a development benchmark for filmed-glass paired data; (2) two failure axes — photometric-level non-identifiability and occlusion-region content loss; (3) a unified-recipe control experiment that bounds what training-time regularization can achieve; (4) boundary results including LightRR fine-tuning and matched-compute students. **No method-superiority or SOTA claim.**
- Build: pdfLaTeX; the official class files from the ICME 2026 author kit ship in `texinputs/` (`IEEEtran.cls`, `IEEEbib.bst`); see `README.md` inside the directory.
- Verification: clean-directory build gives 6 pages (main) and 4 pages (supplement), 0 errors, 0 overfull boxes, 0 undefined citations.

## `cvpr2027_retracted_draft/` — archived historical draft (do not cite)

- Title: *What Single-Image Reflection Removal Does to the Occluder Region: A Diagnostic Benchmark on Tinted Glass*
- Status: **archived for provenance only**. It contains claims (+0.50 dB regularizer benefit, its CI and win-count) that were **retracted** after the unified-recipe, 3-seed control experiment.
- See `README_RETRACTED.md` in that directory before reading or reusing anything from it.
- Build: CVPR author kit is bundled (`cvpr.sty`, `ieeenat_fullname.bst`); body 6 pages + 1 page references (7 total). No supplement.

## Anonymity note

Both drafts are written in anonymous-review style (`cvpr` review mode / ICME anonymous style). The public repository itself is not anonymous; if a venue's double-blind policy forbids public preprints, do not treat this directory as a cleared preprint.

## Figures

Only figures referenced by the frozen ICME draft are shipped (`icme2027/fig/teaser_2rows.jpg`, `icme2027/fig/qual_omega.jpg`). A third figure that displayed retracted confidence intervals is deliberately **not** included. The CVPR draft ships the figure set it was frozen with.
