# Paper drafts

Working drafts of the research paper, kept side by side so the history of the design is
visible. Nothing here is a finished or submitted paper.

| File | What it is | Matching code |
|---|---|---|
| `draft_v1_design_v1.pdf` | First draft (7 October 2026): literature review, hypotheses and Methods for **design v1** | git tag `design-v1` |
| `draft_v2_pilot_methods_as_received.pdf` | Second draft (9 October 2026): same Introduction and Background; Methods rewritten to describe the **design-v2 pilot** exactly as run; pilot figures added. Kept unedited | git tag `design-v2` |
| `draft_v2_corrections.md` | Corrections to the second draft, each with replacement text | current `main` |

The repository's code is design v2. To see the code exactly as it was for design v1, run
`git checkout design-v1`.

## What changed between draft 1 and draft 2

Draft 1 described a study that had not been run. Its pilot (N = 4, 576 configurations)
exposed design problems; draft 2 describes the redesigned study and its pilot. The reasons
for every change are in [`docs/DESIGN_V2.md`](../../docs/DESIGN_V2.md).

| Item | Draft 1 (design v1) | Draft 2 (design v2) | Why |
|---|---|---|---|
| Washout | 500 steps | 2,000 steps | 29% of reservoirs had not forgotten their initial state after 500 steps |
| Hamiltonians | Drawn separately for each channel | The two channels at a grid point share the same realisations | Removes random differences from channel comparisons |
| Clusters for statistics | Grid cell (N, channel, γ, h, W) | Site (N, γ, h, W) | Follows from the shared Hamiltonians |
| Channel comparisons | All surviving configurations | Only sites where ≥ 80% of realisations pass the echo-state check under both channels | The channels were otherwise compared on different parts of the grid |
| Controls | Purity, entropy, log γ, h, W, gap ratio, N | log γ, log h, log W, gap ratio, channel, size indicators; purity only as a sensitivity analysis | Purity and entropy are state diagnostics themselves and absorbed the effect being tested |
| Shot noise | Independent for each observable | All observables at one time computed from the same simulated shots | Matches a real measurement |
| Significance level | 0.05 for each hypothesis | 0.05/3 for each (98.3% and 96.7% intervals) | Three confirmatory tests |
| H1a null | "Gain below 0.02" | "Gain ≤ 0", with 0.02 the smallest gain of interest | The stated null did not match the test |
| Transfer ratio (H1c) | R² of unchanged predictions | r² after a linear recalibration | Δ has a different mean and spread on each task |
| Mackey–Glass | One step ahead | 17 steps ahead | One-step prediction was nearly linear |
| Ridge penalty range | 10⁻¹⁰ to 1 (21 values) | 10⁻¹⁰ to 100 (25 values) | Up to 19% of fits sat at the upper limit |
| System sizes | Main N = 4, 6; held-out N = 8 | Main N = 4, 5, 6; held-out N = 7 | N = 8 needed about 4,900 CPU-hours |
| Gap ratio | "Of H" | Within parity sectors; regime labels from an N = 10 reference ensemble | H has a parity symmetry; N = 4 spectra are too small to classify |
| Description of [20] (Palacios et al.) | Pure-state input, discrete Pauli channels | Mixed-state input diag(s, 1 − s), continuous Lindblad noise | Corrected against the paper |
| Status of results | None | Pilot at N = 4 only; main sweep not run | — |
