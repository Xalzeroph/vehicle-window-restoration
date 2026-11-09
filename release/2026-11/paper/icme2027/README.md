# ICME 2027 submission — source package (work in progress)

**Status: research draft, NOT submitted, NOT camera-ready.** Source frozen on
**2026-11-06** for archival in the public research repository. This is the
**anonymous double-blind copy** (no author names or affiliations in the sources).

## Paper at a glance

- Title: *What Filmed Glass Does to Single-Image Reflection Removal: A Development
  Benchmark and a Diagnostic Study*.
- Venue / template: IEEE ICME 2027 (target); official ICME 2026 author kit ­—
  `IEEEtran.cls` v1.8b (conference option) + `IEEEbib.bst`. Page rule assumed from the
  ICME 2024–2026 official wording: **6 pages including all text, figures and
  references** (the 2027 CFP has not been published yet).
- Abstract: 135 words (ICME limit 100–150). Author block: `Anonymous ICME submission`.
- Positioning: a **development benchmark + diagnostic study**, plus a
  **controlled-intervention protocol** for a training-time regularizer. The regularizer
  section is not a proposed new method; no PSNR/SOTA claim is made.

## Directory contents

```
main.tex                        main document (IEEEtran conference, anonymous)
sec0_abstract.tex               abstract + keywords
sec1_intro.tex                  introduction, Fig.1 (teaser)
sec2_related.tex                related work
sec3_imaging.tex                filmed glass: imaging, capture, development panel
sec4_diagnosis.tex              where current methods fail, Fig.2 (Ω/annulus)
sec5_method.tex                 controlled-intervention protocol (L_quiet)
sec6_experiments.tex            unified-recipe control experiment, LightRR motivation,
                                frozen held-out protocol, negative results
sec7_discussion.tex             scope of evidence, physical boundary, claims
sec8_conclusion.tex             conclusion
refs.bib                        28 verified BibTeX entries (26 cited and printed)
supplementary/supp.tex          standalone supplementary material, S1–S8
fig/teaser_2rows.jpg            Fig.1 teaser (capture pairs)
fig/qual_omega.jpg              Fig.2 qualitative Ω/annulus panels
texinputs/IEEEtran.cls          official ICME 2026 kit class (IEEE)
texinputs/IEEEbib.bst           official ICME 2026 kit BibTeX style (IEEE)
build.ps1                       Windows build script (main + supplementary)
```

No PDFs are bundled; compile locally. Datasets, checkpoints, code, the internal
measurement harness and the internal change log are **not** part of this snapshot.

## Build

Requirements: `pdflatex` (TeX Live 2023+ or TinyTeX) and, for the main text, `bibtex`.
The two IEEE class/style files are bundled in `texinputs/` and are exposed to TeX by
`build.ps1`.

Windows (recommended):

```powershell
cd paper\icme2027
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

POSIX / manual:

```bash
cd paper/icme2027
export TEXINPUTS="$(pwd)/texinputs:${TEXINPUTS}"
export BSTINPUTS="$(pwd)/texinputs:${BSTINPUTS}"
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supp
```

Expected output (verified in a clean temporary directory, 2026-11-09):

- `main.pdf`: **6 pages**, US Letter, references included (2 pdflatex passes + bibtex +
  2 pdflatex passes; 0 errors, 0 overfull boxes, 0 undefined citations).
- `supp.pdf`: **4 pages** (standalone; do not `\input` it into `main.tex`).
- `build/` is disposable and can be ignored/deleted; `build.ps1` recreates it.

## Numbers, scope and limitations

- All quantitative numbers come from the **unified-recipe evaluation protocol** and the
  15-scene **development panel** (1280×832, one ground-truth frame per scene,
  hand-annotated occlusion region Ω), unless a sentence explicitly labels a number as
  historical.
- The panel is a development benchmark, not a hidden test set or a public leaderboard;
  baselines are re-evaluated under our protocol and are not directly comparable to
  published tables. No public state-of-the-art claim is made.
- Known limitations: one ground-truth frame per scene (median 374 frames between the
  input and its ground truth); minimum detectable effect 0.26 dB on the paired panel;
  run-level reproducibility spread is at least as large as the seed-level spread; two
  qualitative panels were not re-annotated in the last Ω pass; captured with a
  hand-held single-frame rig, so no temporal/multi-frame claim is made.
- **Withdrawn numbers.** An old comparison between two pre-unification training
  scripts (+0.50 dB) appears once in the main text, in §6.3, explicitly as a
  *historical observation* — it is not a result and must not be cited as one. The
  supplementary material repeats that number only in historical/limitation notes
  (S3, S8). The previously withdrawn per-scene ΔPSNR confidence interval and the
  withdrawn region-specificity claims are not part of this draft; the bundled figures
  contain no numbers.
- Reference count: `refs.bib` has 28 verified entries, of which 26 are cited and
  appear in the printed reference list (2 spare/optional entries are kept for the
  extended version).

## Data and code

Placeholders to be filled at release time:

- Dataset (Hugging Face): `[DATASET_REPO_URL — to be filled]`
- Code and checkpoints (GitHub): `[CODE_REPO_URL — to be filled]`

Until those links are live, materials are available from the authors **on request**
(for review). No raw capture data, weights or PDFs are stored in this directory.

## Provenance and licensing

- `texinputs/IEEEtran.cls` and `texinputs/IEEEbib.bst` are redistributed unmodified
  from the official ICME 2026 author kit (IEEE; IEEE template licence applies). They
  are bundled only so the snapshot compiles out of the box.
- Paper text and figure images: © the authors; figure images are derived from our own
  captures and are re-used by the supplementary material.
- `main.tex` keeps a `\graphicspath` fallback to `../cvpr2027/fig/` from the internal
  build tree (harmless here; kept to preserve the frozen source).
- Repository note: this project's `.gitignore` ignores `*.png` / `*.jpg` globally, so
  the two figure files must be added explicitly (`git add -f fig/*.jpg`).

## Change history

Snapshot frozen 2026-11-06 after the Lead's independent page verification: main
**6 pages** including references, supplementary **4 pages**, 0 errors, 0 overfull boxes,
0 undefined citations. The internal Plan-B change log and the page-budget measurement
harness live outside this public snapshot; on request they can be shared for
reproducibility.
