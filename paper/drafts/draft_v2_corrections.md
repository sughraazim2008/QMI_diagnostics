# Corrections to draft 2 (`draft_v2_pilot_methods_as_received.pdf`)

Each item gives the place in the draft, what is wrong, and replacement text that matches the
code and the pilot output in `results/pilot2/` and `results/repro14/`. Items 1–14 can be
applied as they stand. Items 15–18 need the author to check something first.

## Front matter

**1. Affiliation footnote.** The footnote reads "Undergraduate research project,
Computational Physics, University of Edinburgh", which presents the work as a University
project. Replace the footnote's first sentence with:

> The author is an undergraduate student at the University of Edinburgh. This work was
> carried out independently; it is not a University project and has not been reviewed or
> endorsed by the University.

Keep the affiliation line only if the target journal accepts a student's institution as an
address for independent work; otherwise use "Independent researcher".

**2. Contact email.** The draft gives emaraat.india@gmail.com; the repository commits use
azimsughra@gmail.com. Use one address in both places.

**3. Abstract.** "capacity-matched, tuned echo state networks" → "tuned echo state networks
of equal readout dimension". The ESN is matched on the number of readout weights, not on
measured capacity.

## Introduction and tables

**4. Contribution 3, Table 1 (Transfer row), Table 2 (H1c), Discussion note.** Replace
"N = 8" by "N = 7" in all four places, and "N ≤ 8" by "N ≤ 7". The redesigned study holds out
N = 7.

**5. Table 1.** "Predicts the gap Δ to a tuned, capacity-matched ESN" → "Predicts the gap Δ
to a tuned ESN of equal readout dimension".

**6. Table 2 (hypotheses).** The table is the design-v1 version and contradicts Sec. IV I.
Replace it with:

| ID | Alternative hypothesis | Null hypothesis | Test |
|---|---|---|---|
| H1a | Adding QMI to a model of Δ containing negativity, coherence and controls raises cross-validated R² | H0a: the gain is not positive | Nested models, cross-validation grouped by site. Supported if the 98.3% bootstrap interval of the gain lies above 0 and the gain is at least 0.02; redundancy claimed if the 96.7% interval lies inside ±0.02 |
| H1b | The QMI–Δ slope differs between pure dephasing and amplitude damping | H0b: no channel × QMI interaction | Interaction term on common-support sites, errors clustered by site, Holm-corrected across QMI variants at α = 0.05/3; channel independence claimed if a TOST places the standardised interaction inside ±0.1 |
| H1c | A model fitted in one setting keeps at least 70% of its explained variance in a held-out channel, a held-out size (N = 7) and a second task | H0c: the transfer ratio is below 0.70 in at least one held-out setting | Transfer ratio = r²(target, after linear recalibration) / R²_cv(source); H0c rejected when the lower 98.3% bootstrap bound is ≥ 0.70 in every transfer |

Caption: "Hypotheses, null hypotheses and tests. The margins (0.02 in R², ±0.1 in the
standardised interaction, 0.70 in the transfer ratio) are pre-specified and will be fixed in
the pre-registration."

**7. "Pre-registered".** Nothing is pre-registered yet. In the captions of Table 2, Fig. 4
and Fig. 5, replace "pre-registered" by "pre-specified".

## Methods

**8. Sec. IV A, gap ratio.** After "regime labels … are assigned per (h, W) point from a
reference ensemble of 100 realisations at N = 10", add:

> The gap ratio is computed separately within the two sectors of the parity
> P = ∏ᵢ σᶻᵢ, which commutes with H, and averaged [42]; on the full spectrum the two
> independent level sequences would mimic Poisson statistics in every regime.

**9. Sec. IV D, Mackey–Glass.** "one-step Mackey–Glass prediction (τ = 17)" is wrong for the
pilot. Replace with:

> and Mackey–Glass prediction (delay τ = 17) 17 steps ahead, i.e. one delay time. At this
> horizon a regression on the last ten inputs and their products reaches an NRMSE of only
> 0.14, so the task needs memory.

Also correct "scored on the same reservoir trajectories": the three NARMA orders share one
trajectory; Mackey–Glass uses a separate drive with its own diagnostics window.

**10. Sec. IV F, add after Eq. (7):**

> Because the ESN is matched on readout dimension, its error is a single number per task
> and system size. Within one system size Δ therefore equals the quantum reservoir's NRMSE
> up to a constant, and models of Δ are models of quantum-reservoir error; the comparison
> with the ESN contributes the sign of the gap.

**11. Sec. IV G, controls.** "log γ, h, W, gap ratio r̄, channel" → "log₁₀ γ, log₁₀ h,
log₁₀ W, gap ratio r̄ and channel (and one indicator per system size in the full study)".
Cite the diagnostics where they are defined so that the references are used: negativity
[43], ℓ1 coherence [44], and the pure-state identity between mutual information and
entanglement entropy [45] if that sentence is kept; otherwise remove [45] from the list.

**12. Table 4, "Reproduction of [20]" column.** "As in [20]" is not accurate. Replace the
column with: N = 5; bit flip and phase flip; error probability p_err = 0 to 0.05 per input
period; h = 1; W = 0 (ergodic) and 10 (localised); 5 realisations; outputs IPC (to degree 4
and delay 12, 10⁴ steps), coherence, negativity. Add to the caption: "The reproduction is
smaller than the original study (degree 6, 10⁵ steps) and uses 15 readout features, so it is
qualitative."

**13. Sec. IV J and Fig. 1 caption, reproduction of [20].** The caption says IPC "rises
monotonically with coherence and with QMI". That is stronger than the run supports. Replace
the caption's second sentence with:

> At an error probability of 0.05, negativity has fallen to 21–26% of its noiseless value
> while total IPC remains at 92–94%, and the bit-flip and phase-flip runs fall on a common
> curve, as in Ref. [20]. The group means of IPC and coherence fall together, but with 15
> readout features IPC sits at its ceiling until p_err ≈ 0.02, so the monotonic
> IPC–coherence relation of Ref. [20] is only weakly resolved here (Spearman 0.45 across
> runs). Without readout noise, IPC rises with decoherence strength; with it, IPC falls,
> reproducing the reversal reported in Ref. [20].

In Sec. IV J, replace "a qualitative reproduction of Ref. [20]'s IPC–coherence and
IPC–negativity relations" by "a qualitative, reduced-size reproduction of the main findings
of Ref. [20] (Fig. 1)".

**14. Sec. IV, opening paragraph.** The paragraph compares "the original draft", "design v1"
and a "companion version". This is useful in a working draft and must be removed before
submission; a paper describes one study. One statement in it and in Sec. IV G is also
inaccurate: draft 1 already divided T by N for comparison across sizes, so "QMI not
normalised per qubit" is not a difference between the designs.

## Figure captions

**Fig. 2.** Two statements are contradicted by the figure itself. Replace everything after
"per site." with:

> Under pure dephasing all three diagnostics fall with γ. Under amplitude damping coherence
> falls, but negativity and mutual information are non-monotonic and rise again at strong
> damping. NRMSE generally rises with γ;
> the exception is pure dephasing in the localised regime, where it is roughly flat. The
> two channels therefore separate the diagnostics from performance in different ways, which
> is the question H1b addresses.

**Fig. 3.** Delete "(Table in the companion Discussion)" and change "(Sec. IV I)" to
"(Fig. 4)". The sentence then reads: "Note that Δ is never positive in this pilot (0 of 498
configurations), so …".

**Fig. 4.** The figure contains rows that add coherence, negativity or all three diagnostics,
not only QMI. Replace the first sentence with: "Gain in cross-validated R² from adding a
predictor to the nested model: QMI in ten of the rows (pooled, by subset and under each
sensitivity variant) and, for comparison, coherence, negativity and all three diagnostics
in the other three." Replace the last sentence with: "Every point estimate lies inside the ±0.02
equivalence region and every interval includes zero. For the primary test the 96.7% interval
(−0.011 to +0.009) lies inside the region, so at pilot scale QMI is redundant with negativity
and coherence."

**Fig. 5.** "By the pre-registered rule" → "By the pre-specified rule". The numbers in the
caption are correct.

**New figure for H1b.** The pilot's one positive result has no figure. Add
`results/pilot2/analysis/figures/h1b_channel_slopes.png` with the caption:

> Pilot data. Standardised slope of the gap Δ on QMI under each channel, with 95% intervals
> from site-clustered standard errors, on the 44 common-support sites (352 configurations).
> The channel × QMI interaction is +0.33 (SE 0.06, Holm-adjusted p < 0.001) for total mutual
> information and +0.46 (SE 0.12, p = 0.0005) for pairwise QMI. At pilot scale the
> relationship between QMI and performance depends on the decoherence channel.

## For the author to check

**15. Reference [17].** The title in draft 2, "Exploring quantum mechanical advantage for
reservoir computing", differs from draft 1, "Exploring quantumness in quantum reservoir
computing" (Götting, Lohof and Gies, Phys. Rev. A 108, 052427; arXiv:2302.03595). Open the
paper and use its actual title.

**16. Reference [26].** Draft 1 lists Ricci et al. as published (npj Quantum Information,
doi:10.1038/s41534-026-01229-8); draft 2 says it is still a preprint. One of them is wrong.

**17. Reference [41].** Draft 2 cites PRX Quantum 5, 040325 (2024) for Kobayashi, Fujii and
Yamamoto; the earlier check could not confirm a journal version. Confirm on the PRX Quantum
site or cite arXiv:2406.15783.

**18. Reference [45].** "10th ed." → "10th anniversary edition".

**19. Authorship statement.** "All design decisions, physics checks and reported numbers were
reviewed by the author" is to be kept only once that is the case.

**20. The set-up of the reproduction** (item 12) was taken from a summary of
arXiv:2409.17734. Check h, W, the coupling distribution, Δt and the noise definition against
the paper before describing the run as a reproduction of it.
