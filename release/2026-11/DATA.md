# Data card — filmed-glass November 2026 batch (`nov02`)

**Hosting:** Hugging Face dataset `zmy1234567890/filmed-glass-nov02` (private during pre-release review; access on request). Large binaries are intentionally not in this git repository.

## What was captured

- 15 capture sessions, all on **one day, one location, one filmed glass pane**. The only intended variation across sessions is a **rotation of the pane about the optical axis**; metadata does **not** record the rotation angle (so angle labels are unavailable and angle-conditioned claims are not made).
- Each session: a short video sequence. Input frames are frames with reflections; the ground-truth frame is the same scene with the reflection path blocked by a black cloth (same camera position, after the input).
- Ground truth is available once per session (15 GT frames), not per input frame.

## Files

```
I/        1499 PNG  (936 x 590)   input frames with reflections
gt/         15 PNG                black-cloth ground-truth transmission frames
R/        1499 PNG                AUXILIARY pseudo reflection layer (estimated; NOT ground truth)
omega/    1499 PNG                AUXILIARY pseudo occluder/reflection masks (heuristic; NOT ground truth)
meta.json 1499 records            per-pair metadata
qc/                              optional QC artifacts (when generated)
```

`meta.json` fields (per pair): `domain` (`nov02a` indoor / `nov02b` outdoor), `domain_desc`, `scene`, `src` (frame file), `gt_frame`, `gt_src`, `g` (median `I/T` photometric scale), `gain_dev`, `shift` (registration shift), `omega_frac`, `omega_src` (currently `random_disk`), `R_abs_mean`, `size`.

## Ground truth semantics

- **`gt/` is the only ground truth.** It is a black-cloth capture: reflection removed, illumination changed by the cloth (documented in the paper).
- **`R/` and `omega/` are pseudo-labels**, produced by the released data-build code. They are useful as auxiliary supervision or for sanity checks, but must **not** be described or evaluated as ground truth. The `omega/` masks for this batch are interior random discs, not hand-annotated occlusion footprints.
- There is **no hand-annotated occlusion mask** in this batch; therefore region-restricted PSNR inside the occlusion region (`PSNR_Ω`) is *not* computed on this batch. Region claims require the 15-scene 1019 panel with hand masks.

## Intended use

- Out-of-batch evaluation of models trained on other captures (with provenance exclusions: models trained or selected on this batch are excluded from evaluation; see `docs/nov02_report_v3.md`).
- Diagnostics: robust behavior of identity vs. zero-shot vs. fine-tuned models; photometric-scale statistics; local-content unidentifiability analysis.

## Limitations (do not over-claim)

- Small effective sample size: 15 sessions cluster into **5 unique scene contents** under same-place near-duplicate clustering (11 under strict near-duplicate clustering). Cluster-level MDE is ≈1.8 dB; per-image differences of ~1 dB are not significant.
- Mostly weak reflections (median photometric scale ≈1.01), one device, one location, one pane, one film.
- No angle labels, no per-frame GT, no hand occlusion masks.
- The batch was **not** collected for training: mixing it into training reduced accuracy in a controlled study (`docs/results_summary_2026-11.md`).

## Privacy / license

Frames may contain window views and interiors. Redistribution is research-only, and released copies are privacy-reviewed (faces/plates reviewed, blurring applied where needed). Do not attempt to identify persons. See the dataset card on Hugging Face for the final license and the privacy manifest.
