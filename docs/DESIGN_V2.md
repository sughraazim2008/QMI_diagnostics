# Design v2: what changed and why

Design v1 is the design of paper draft v1. The pilot of 7 October 2026
([PILOT_RESULTS.md](PILOT_RESULTS.md)) found six problems in it. Design v2 fixes them. This
document lists every change with its reason, restates the hypotheses so that each one matches
its statistical test, and lists what is still open. The code implements v2;
[METHODS.md](METHODS.md) describes it in paper form.

A note on what a redesign can do. It can make the hypotheses precise, make each test capable
of giving any of its possible answers, and remove artefacts that would make an answer
untrustworthy. It cannot make the alternative hypotheses come out true. In the pilot the
evidence pointed towards the null for H1a, and v2 is built so that such a result is a clean,
publishable finding and a positive result, if it appears, is believable.

## 1. Research question

In small spin-network quantum reservoirs under local decoherence, read out with finite-shot
noise, does the quantum mutual information (QMI) of the input-driven state, computed without
task targets, improve out-of-sample prediction of NARMA-10 performance relative to a tuned
echo state network of equal readout dimension, beyond what negativity and l1-norm coherence
predict? Does that relationship differ between pure dephasing and amplitude damping, and does
it carry over to another channel, a larger system and other tasks?

## 2. Hypotheses

Each hypothesis has a null, an alternative, one test statistic and a three-way decision.
The three are tested at α = 0.05/3 = 0.0167 each (Bonferroni), so the chance of any false
confirmation across the study is at most 5%.

| | Null | Alternative | Statistic | Supported if | Equivalence (a positive finding of "no effect") if |
|---|---|---|---|---|---|
| **H1a** | H0a: adding QMI does not raise cross-validated R² (gain ≤ 0) | Gain > 0 | Gain = R²_cv(M₁) − R²_cv(M₀) | 98.3% interval above 0 **and** point estimate ≥ 0.02 | 96.7% interval inside (−0.02, +0.02): QMI is redundant |
| **H1b** | H0b: the QMI slope is the same in both channels (interaction = 0) | Interaction ≠ 0 | Standardised channel × QMI coefficient | Holm-adjusted p < 0.0167 | TOST places the interaction inside ±0.1: channel-independent |
| **H1c** | H0c: in at least one transfer the model keeps less than 70% of its explained variance | Ratio ≥ 0.70 in every transfer | Transfer ratio = r²_target / R²_cv,source | Lower 98.3% bound ≥ 0.70 in all transfers | Not applicable; report each ratio with its interval |

Anything else is reported as inconclusive. 0.02 is the smallest gain treated as practically
relevant and ±0.1 the largest interaction treated as negligible; both are judgement calls
that must be justified in the pre-registration and not moved afterwards.

Corrections relative to draft v1:

* Draft v1 stated H0a as "the gain is below 0.02" but tested "lower bound above 0". These are
  different tests. v2 states the null as "gain ≤ 0" and uses 0.02 as the smallest effect of
  interest, which is what the decision rule always did.
* H1c needs every transfer to pass, so its null is "at least one fails". No correction across
  transfers is needed for that kind of test.
* A descriptive question is added that is not a hypothesis: in what share of configurations
  is the gap positive at all? It is reported with a confidence interval at each system size.

## 3. Changes

### Validity (is the study measuring what it claims?)

| # | Problem in v1 (evidence) | Change in v2 |
|---|---|---|
| 1 | 29% of reservoirs had no fading memory after 500 steps, mostly under dephasing, so the channels were compared on different subsets | Washout raised to 2,000 steps. Measured pass rates at N = 4: amplitude damping 86% → 97%, dephasing 57% → 74% (`results/pilot/esp_vs_washout.csv`) |
| 2 | Channel comparisons were confounded with those exclusions | H1b and the channel transfer use only **common-support sites**: grid points where at least 80% of realisations pass under both channels. The excluded region is reported as a result in its own right |
| 3 | Each channel had its own random Hamiltonians, adding noise to every channel comparison | **Paired design**: both channels at a grid point use the same Hamiltonian realisations. Clusters for cross-validation and the bootstrap are now sites (N, γ, h, W), which are independent of one another |
| 4 | Purity, entropy and log γ were "controls" and absorbed nearly everything the diagnostics could explain (0.709 of 0.715) | Primary controls are only what the experimenter sets (γ, h, W, N, channel) plus the gap ratio. Purity is a state diagnostic like the others and enters in a pre-specified sensitivity analysis. Entropy is dropped (rank correlation −1.00 with purity) |
| 5 | Transfer was scored with R² of unchanged predictions, which no model can pass when the outcome has a different mean and spread per task | Transfer ratio uses r², the variance explained after a linear recalibration (one intercept, one positive scale). Raw R² and Spearman correlation are still reported |
| 6 | Mackey–Glass one-step prediction was nearly linear (a 10-lag formula scored 0.0005) | Horizon of 17 steps, one delay time: the 10-lag formula scores 0.14 and a reservoir 0.006, so the task needs memory |
| 7 | N entered as a linear control | One indicator per system size, which absorbs the size-specific ESN baseline exactly. Transfer models carry no size term, so they can be applied to a held-out size |
| 8 | Three confirmatory tests at 0.05 each | Bonferroni: α = 0.0167 each |

### Accuracy (are the numbers right?)

| # | Problem in v1 | Change in v2 |
|---|---|---|
| 9 | Shot noise was added independently to each observable, although ⟨σᶻ⟩ and ⟨σᶻσᶻ⟩ come from the same z-basis shots | **Same-shot sampling**: at each readout time the outcome frequencies are sampled once and all observables are computed from that sample, giving the correct variances and correlations |
| 10 | 8% of NARMA-10 and 19% of NARMA-20 ridge fits chose the largest penalty | Penalty range extended from 10⁻¹⁰–1 to 10⁻¹⁰–100 (25 values) |
| 11 | The tuned ESN's input scaling for NARMA-20 sat at the edge of its search range | Search range extended down to 10⁻³ |
| 12 | Gap ratio computed "of H" | Computed within parity sectors (already in the v1 code) |

### Reliability (would a repeat give the same answer?)

| # | Problem in v1 | Change in v2 |
|---|---|---|
| 13 | No statement of how reproducible one configuration's score is | The report gives the reliability of the outcome: the share of between-configuration variance that is not sampling error over input realisations |
| 14 | Two system sizes, the smaller of which (N = 4) has no usable regime structure | Main sweep uses N = 4, 5 and 6; results are also reported per size |
| 15 | Held-out size N = 8 needed about 4,900 CPU-hours | Held-out size is N = 7 with 3 input realisations and 4 Hamiltonian realisations per site |
| 16 | One seed scheme per channel | Every random draw still derives from one base seed; reruns are bit-for-bit identical (tested) |

## 4. What v2 does not change

* **The outcome within one system size is still NRMSE with a constant subtracted.** The ESN
  is matched on readout dimension, as the fair-comparison literature requires, so its error
  is one number per (task, N). H1a and H1b therefore test whether QMI predicts the quantum
  reservoir's error; the comparison with the ESN adds the *sign* of the gap and the
  comparison across sizes. The paper should say this in one sentence. A baseline that varies
  by configuration (for example an ESN matched to each reservoir's measured capacity) would
  change the question and is left as a possible extension.
* **Localised reservoirs under pure dephasing mostly lack fading memory even after 2,000
  steps** (W = 20: 39% pass). This is a property of the physics, not of the
  tolerance: the failures have trace distances around 10⁻², the passes around 10⁻¹⁴. v2
  reports it and excludes those sites from channel comparisons instead of hiding it.
* **Collinearity.** QMI, coherence and negativity are strongly correlated (0.86–0.92 in the
  pilot). No design removes that; it is the substance of H1a.
* **Small systems.** N ≤ 7 is far from the sizes where a quantum advantage is conjectured.
  Conclusions are limited to these sizes, as in [32].

## 5. Before pre-registering

1. Read the design-v2 pilot in [PILOT_RESULTS.md](PILOT_RESULTS.md), Section 6, and confirm
   the fixes behave as intended.
2. Decide and justify the two margins (0.02 and 0.1) and the 0.70 threshold.
3. Decide whether purity belongs in the primary controls (v2 says no) and write down why.
4. Check the compute budget for the main and transfer sweeps (`scripts/estimate_cost.py`).
5. Freeze the code (git tag), then register hypotheses, design and analysis plan on OSF.
6. Only then run `scripts/run_sweep.py main` and `transfer`.
