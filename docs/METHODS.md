# Methods (as implemented, design v2)

This is Section 2 of the paper rewritten to match the code in this repository exactly.
It describes design v2; [DESIGN_V2.md](DESIGN_V2.md) lists what changed from draft v1 and why.
Each subsection names the module that implements it. Values marked † are proposals to be
fixed in the pre-registration. Where the code departs from draft v1 of the paper, the
change is listed in [DESIGN_V2.md](DESIGN_V2.md).

## 2.1 Reservoir model — `hamiltonian.py`, `reservoir.py`

The reservoir is a network of N qubits with all-to-all transverse-field Ising couplings and
on-site disorder [5], [7], [14]:

$$H = \sum_{i<j} J_{ij}\,\sigma^x_i \sigma^x_j + \frac{1}{2}\sum_{i=1}^{N} (h + D_i)\,\sigma^z_i,
\qquad J_{ij} \sim \mathcal{U}\!\left[-\tfrac{J_s}{2}, \tfrac{J_s}{2}\right], \quad D_i \sim \mathcal{U}[-W, W].$$

Energies are in units of J_s = 1 and times in units of 1/J_s (ħ = 1). A *site* is one grid
point (N, γ, h, W). Each site draws its own set of Hamiltonian realisations, and the two
channels at a site use the same realisations (a paired design), so sites are independent of
one another and channel differences are not blurred by different random couplings.

**Dynamical regime.** H commutes with the parity P = ∏ᵢ σᶻᵢ, because every coupling term flips
two spins. The mean adjacent-gap ratio ⟨r⟩ [Oganesyan and Huse 2007; Atas et al. 2013] is
therefore computed separately in the two parity sectors and averaged; computed on the full
spectrum it would mix two independent level sequences and read as Poisson even for ergodic
dynamics. Reference values are 0.531 (GOE) and 0.386 (Poisson). Because the sectors of N = 4
and N = 6 hold only 8 and 32 levels, ⟨r⟩ of a single small Hamiltonian is too noisy to
classify a regime. Regime labels are therefore assigned per (h, W) point from a reference
ensemble at N = 10 (100 realisations, `scripts/pilot_gap_ratio.py`): *ergodic* if
⟨r⟩ ≥ 0.50, *localised* if ⟨r⟩ ≤ 0.45 and W ≥ 10, *other* otherwise. The per-configuration
gap ratio is still recorded and entered as a control.

**Input injection.** At step k a scalar input s_k ∈ [0, 1] is written into qubit 1 by the
reset-and-write map [5]:

$$\rho \;\mapsto\; |\psi_{s_k}\rangle\langle\psi_{s_k}| \otimes \mathrm{Tr}_1(\rho),
\qquad |\psi_{s}\rangle = \sqrt{1-s}\,|0\rangle + \sqrt{s}\,|1\rangle ,$$

followed by evolution for one input period Δt under the master equation of Section 2.2.
Two further encodings are implemented: the mixed state ρ₁ = diag(s, 1 − s), which is the
encoding of the main text of [14] and is used for the reproduction, and the rotation
|ψ_s⟩ = exp(−iπsσˣ/2)|0⟩, used as the sensitivity check.

## 2.2 Decoherence channels and simulation — `channels.py`, `reservoir.py`

Decoherence acts locally on every qubit through a Lindblad master equation:

$$\dot{\rho} = -i[H, \rho] + \sum_{i=1}^{N} \left( L_i \rho L_i^{\dagger} - \tfrac{1}{2}\{L_i^{\dagger} L_i, \rho\} \right).$$

| Channel | Jump operator | Unital | Kraus parameter after time t | Role |
|---|---|---|---|---|
| Pure dephasing | L = (√γ/2) σᶻ | Yes | p = (1 − e^{−γt/2})/2 | Main comparison (H1b) |
| Amplitude damping | L = √γ σ⁻, σ⁻ = \|0⟩⟨1\| | No | p = 1 − e^{−γt} | Main comparison (H1b) |
| Phase flip, bit flip | L = √γ σᶻ, √γ σˣ | Yes | p = (1 − e^{−2γt})/2 | Reproduction of [14] only |

With these definitions pure dephasing and amplitude damping both damp single-qubit coherences
as e^{−γt/2} (T₂ = 2/γ), so the main comparison is at equal coherence decay. Amplitude damping
additionally relaxes populations at rate γ (T₁ = 1/γ) towards |0⟩.

States are dense 2ᴺ × 2ᴺ density matrices. Each input period is divided into V sub-intervals
of length τ = Δt/V, at the end of which the readout is taken (Section 2.3). Within a
sub-interval:

* **N = 4:** the exact propagator exp(𝓛τ) of the vectorised Liouvillian is applied.
* **N ≥ 5:** n_s second-order (Strang) steps of length δ = τ/n_s are applied:
  half a step of the local channel, the unitary exp(−iHδ), then half a step of the channel.
  Each local channel is the exact solution of its single-qubit dissipator, applied through its
  Kraus form.

n_s is calibrated per channel and γ (`scripts/calibrate_ns.py`, results in `configs/n_s.json`)
as the smallest power of two for which the Strang state stays within trace distance 10⁻⁴ of
the exact propagator after 100 input periods, in the worst case over the (h, W) grid and two
Hamiltonian realisations at N = 4 and N = 5. For N > 5 the calibrated n_s is doubled as a
safety margin. The measured values are in [Section 2.10](#210-validation).

## 2.3 Readout and finite sampling — `reservoir.py`, `readout.py`

At the end of each of the V sub-intervals the expectation values ⟨σᶻᵢ⟩ and ⟨σᶻᵢσᶻⱼ⟩ are
recorded (time multiplexing) [5], [8]. This gives D = V [N + N(N − 1)/2] features plus a bias:
D = 100, 210 and 360 for N = 4, 6 and 8 with V = 10†, Δt = 10†.

Finite sampling is part of every primary analysis. All observables are diagonal in the
computational basis, so at each readout time one batch of n_shots projective z-basis
measurements yields all of them. The simulation samples the outcome frequencies p̂ of that
batch once and computes every observable from the same sample. The frequencies are drawn from
the Gaussian limit of the multinomial distribution, p̂ = p + (√p ∘ z − p (√p · z))/√n_shots
with z standard normal, which has the exact mean p and the exact covariance
(diag p − ppᵀ)/n_shots. Each observable therefore has variance (1 − ⟨O⟩²)/n_shots and the
errors of different observables carry the correlations of the state. Noise is independent
between time steps and is present in training, validation and test data alike. The primary
setting is n_shots = 10⁶, close to the σ = 10⁻³ of [14]; 10⁵ and 10⁴ are secondary.
Measurement back-action is not modelled: an ensemble or restart protocol is assumed [28],
which is discussed as a limitation.

## 2.4 Tasks and capacity — `tasks.py`, `ipc.py`

**NARMA-10** [3] is the primary task. With u_k drawn independently from 𝒰[0, 0.5]:

$$y_{k+1} = 0.3\,y_k + 0.05\,y_k \sum_{i=0}^{9} y_{k-i} + 1.5\,u_{k-9}\,u_k + 0.1 .$$

The reservoir receives s_k = 2u_k and the readout at step k is trained on y_{k+1}, which
depends only on inputs up to u_k. A realisation is redrawn if any NARMA series diverges on
it; the number discarded is stored in the result table. Each realisation has a washout of
2,000 steps, set by the echo-state pilot (Section 2.10), then 2,000 training, 500 validation
and 1,000 test steps†. Five input realisations are
shared by all configurations and by the classical baseline.

**Transfer tasks.** NARMA-5 uses the same equation with the sum over five terms and the
product u_{k−4}u_k. NARMA-20 uses the bounded variant of Rodan and Tiňo (2011),
y_{k+1} = tanh(0.3 y_k + 0.05 y_k Σ_{i=0}^{19} y_{k−i} + 1.5 u_{k−19} u_k + 0.01). All three
NARMA orders are scored on the same reservoir trajectory, since they share the input.
**Mackey–Glass** [ref to add: Mackey and Glass 1977] uses ẋ = 0.2 x(t − 17)/(1 + x(t − 17)¹⁰) − 0.1 x,
integrated with a fourth-order Runge–Kutta scheme (step 0.1) and sampled every time unit after
a transient of 1,000; the input is s_k = (x_k − 0.2)/1.2 and the target is x_{k+17}, one
delay time ahead. At this horizon a regression on the last ten inputs and their products
reaches an NRMSE of only 0.14, so the task cannot be solved without memory; one-step
prediction, by contrast, is almost linear.

**Information processing capacity** follows [4]. Targets are products of normalised Legendre
polynomials of delayed inputs u = 2s − 1, up to total degree 3 and delay 20† (2,023 targets).
The capacity of a target z is ‖Qᵀz‖²/‖z‖², with Q an orthonormal basis of the centred
features, evaluated on the 3,500 post-washout steps of the first input realisation with the
primary shot noise. Capacities not exceeding the largest capacity obtained from a
time-shuffled copy of the input are set to zero. With series this short the threshold is
about 0.04–0.05 per target, so weak capacities are lost and the total is a lower bound; it is
used as a secondary covariate, not as an absolute capacity.

## 2.5 Readout and classical baseline — `readout.py`, `esn.py`

All readouts are ridge regressions with an unpenalised bias, minimising
‖y − Xw − b‖² + λ‖w‖² on the training segment. Features are centred with training means and
not rescaled. λ is chosen from 25 log-spaced values between 10⁻¹⁰ and 10² by validation NRMSE,
and the training-segment fit at that λ is scored on the test segment:

$$\mathrm{NRMSE} = \sqrt{ \langle (y_k - \hat{y}_k)^2 \rangle / \mathrm{Var}(y) } .$$

The classical baseline is a leaky echo state network [1], [43]:

$$x_{k+1} = (1-a)\,x_k + a \tanh\!\left( W x_k + W_{\mathrm{in}}\, s_k + b \right).$$

It has D nodes, so both models fit D + 1 readout weights [32], [33]. W has 10% non-zero
entries drawn from 𝒰[−1, 1] and is rescaled to spectral radius ρ; W_in and b are drawn from
𝒰[−1, 1] and multiplied by the input and bias scalings. The ESN is tuned with a
tree-structured Parzen estimator (Optuna) over a ∈ [0.05, 1] (log scale), ρ ∈ [0.1, 1.5],
input scaling ∈ [10⁻³, 2] (log scale) and bias scaling ∈ [0, 1], with 100 trials per
(task, D)†. The objective is validation NRMSE averaged over the five input realisations and
three tuning seeds. The selected setting is then scored on the test segment with ten fresh
weight seeds. ESN features are noiseless.

The quantum reservoir receives no comparable tuning: each configuration is fixed and only λ
is fitted. Two reference baselines are reported: an untuned ESN (a = 1, ρ = 0.9, unit input
scaling, no bias) and a ridge regression on the last ten inputs and their pairwise products.

## 2.6 Performance gap — `stats.prepare`

For configuration c with readout dimension D_c,

$$\Delta_c = \overline{\mathrm{NRMSE}}_{\mathrm{ESN}}(D_c) - \overline{\mathrm{NRMSE}}_{\mathrm{QRC}}(c),$$

with means over the five input realisations (and the ten weight seeds for the ESN).
Δ_c > 0 means the quantum reservoir is better. Because the ESN is matched on readout
dimension, its error is one number per task and system size; within one size Δ is the quantum
reservoir's NRMSE with a constant subtracted, and every model includes an indicator for each
size. Models of Δ are therefore models of quantum-reservoir error, and the comparison with
the ESN contributes the sign of the gap, which is reported as the share of configurations
with Δ > 0 at each size.

## 2.7 State diagnostics — `diagnostics.py`

Diagnostics are computed from the reservoir state alone. Starting from the washed-out state
at the end of the first input realisation, the reservoir is driven by a fresh sequence of 200
inputs from the task's input distribution (uniform for the NARMA family, a separate
Mackey–Glass segment for that task). Each diagnostic is evaluated on the state at the end of
every period, just before the next reset, and averaged over the window. Entropies are in bits.

| Diagnostic | Definition | Notes |
|---|---|---|
| Negativity N̄ | Mean of (‖ρ^{T_A}‖₁ − 1)/2 over all 2^{N−1} − 1 bipartitions [36] | As in [14]; mean log-negativity log₂‖ρ^{T_A}‖₁ is secondary |
| l1 coherence C | Σ_{i≠j} \|ρ_ij\| / (2ᴺ − 1), computational basis [37] | Basis matches the readout |
| Total mutual information T | [Σᵢ S(ρᵢ) − S(ρ)] / N [14], [38] | Primary QMI |
| Pairwise QMI Ī | Mean over pairs of S(ρᵢ) + S(ρⱼ) − S(ρᵢⱼ) | Secondary QMI; half-chain I(A:B) also stored |
| Classical correlations, hookup | T(Δ[ρ]) and Σᵢ H(diag ρᵢ) − S(ρ) [39] | Stored for the reproduction of [14] |
| Controls | log₁₀ γ, log₁₀ h, log₁₀ W, gap ratio r, channel, one indicator per system size | Entered in every model; a control that is constant in a fitting set is dropped |
| Purity Tr ρ² | | A state property like the diagnostics, so not a primary control; added in a sensitivity analysis. S(ρ) is stored but not used: it is rank-correlated with purity at −1.00 |

For pure states the mutual information across a bipartition equals twice the entanglement
entropy [40], so any difference between QMI and entanglement measures comes from mixedness.
The sensitivity analysis with purity shows how much of QMI's contribution is of that kind.

## 2.8 Experimental design — `config.py`

The unit of analysis is a configuration: one Hamiltonian realisation under one channel at
one site. Sites (N, γ, h, W) are the groups for cross-validation and the clusters for the
bootstrap and for robust standard errors, because the two channels at a site share
Hamiltonians.

| Factor | Main sweep (`main`) | Transfer sweep (`transfer`) | Reproduction of [14] |
|---|---|---|---|
| N | 4, 5, 6 | 7 | 5 |
| Channel | Dephasing, amplitude damping | Same | Bit flip, phase flip |
| γ (units of J_s) | 8 log-spaced values, 10⁻³ to 1† | Every second main value | p_err from 0 to 0.05 |
| h | 0.1, 1, 10 | Same | 1 |
| W | 0.1, 5, 20 | Same | 0, 10 |
| Hamiltonian realisations per site | 10 | 4 | 5 (default) |
| Input realisations | 5 | 3 | 1 |
| Sites | 216 | 36 | — |
| Configurations | 4,320 | 288 | 120 |
| Outputs | NARMA-10, IPC, diagnostics | NARMA-5/10/20, Mackey–Glass, diagnostics | IPC, diagnostics |

The h and W levels were chosen from the gap-ratio pilot ([Section 2.10](#210-validation)).

## 2.9 Statistical analysis — `stats.py`, `scripts/analyze.py`

Configurations that fail the echo-state check are excluded. Predictors are standardised
within each fitting set. Cross-validation is 5-fold, grouped by site, and repeated 10 times;
R²_cv of one repeat is 1 − SSE/SST of the pooled out-of-fold predictions, and repeats are
averaged. Uncertainty comes from a cluster bootstrap over sites (2,000 resamples), recomputing
the full cross-validation in each resample, with percentile intervals. The three confirmatory
hypotheses are each tested at α = 0.05/3 (Bonferroni), so support intervals are 98.3% and
equivalence intervals 96.7%.

**H1a (added value of QMI).** Two linear models of Δ are compared on identical folds:

$$M_0: \ \Delta \sim \bar{N} + C + \text{controls}, \qquad M_1: \ \Delta \sim \bar{N} + C + T + \text{controls}.$$

The statistic is the gain ΔR²_cv = R²_cv(M₁) − R²_cv(M₀). The null hypothesis H0a is that the
gain is not positive. H1a is supported if the 98.3% interval lies above 0 and the point
estimate is at least 0.02†, the smallest gain treated as practically relevant. QMI is declared
redundant if the 96.7% interval lies within ±0.02 (two one-sided tests at α). Otherwise the
result is inconclusive. The test is run pooled, within each channel, within the ergodic and
localised regimes, and within each N. Pre-specified sensitivity analyses swap T for pairwise
QMI, use log-negativity, add purity to the controls, repeat the comparison at 10⁵ and 10⁴
shots, and repeat it with gradient-boosted trees (fewer bootstrap resamples). The same nested
comparison is also run with coherence, with negativity, and with all three diagnostics as the
added predictors, to show what each contributes. Spearman correlations, variance
inflation factors and a commonality analysis show how much explained variance is shared.

**H1b (channel dependence).** Channel comparisons use only *common-support* sites, those
where at least 80% of the realisations pass the echo-state check under both channels, so that
the two channels are compared on the same Hamiltonians and the same part of the grid. On
these data, with Δ and all continuous predictors standardised:

$$\Delta \sim \bar{N} + C + T + \text{controls} + \text{channel} + \text{channel} \times T .$$

The channel × T coefficient (amplitude-damping slope minus dephasing slope) is tested with
standard errors clustered by site and a t reference distribution with (number of sites − 1)
degrees of freedom, Holm-corrected across the two QMI variants and compared with α. Channel-independence is
claimed only if TOST places the interaction within ±0.1†. Interactions of channel with C and
N̄ are reported as secondary.

**H1c (transfer).** M₁ without system-size terms is fitted on a source setting and its
predictions are computed for a target setting with the source's standardisation and
coefficients. Three kinds of transfer are tested: between channels (both directions, on
common-support sites), from N = 4, 5 and 6 to N = 7, and from NARMA-10 to each second task on
the configurations of the transfer sweep. Because Δ has a different mean and spread in each
task and system size, the transfer ratio is r²_target / R²_cv,source, where r is the Pearson
correlation between the source model's predictions and the observed Δ in the target (set to
zero if negative). r² is the variance the source model explains in the target after a linear
recalibration with one intercept and one positive scale. H0c, that at least one transfer
keeps less than 70% of the explained variance, is rejected when the lower 98.3% bootstrap
bound of the ratio is at least 0.70 in every transfer. The R² of the unrecalibrated
predictions and the Spearman correlation are reported alongside.

## 2.10 Validation — `tests/`, `scripts/`

The main sweep starts only after these checks pass.

1. **Unit tests** (`pytest`, 57 tests). A Bell pair gives negativity 0.5 and T = 2 bits;
   |+⟩^{⊗N} gives C = 2ᴺ − 1 and T = 0; product and classically correlated states give zero
   negativity; a GHZ state gives negativity 0.5 on every cut; each channel's Kraus form equals
   the exponential of its Lindblad generator, its Choi matrix is positive semidefinite and
   trace-preserving; amplitude damping relaxes to |0⟩; dephasing and amplitude damping give the
   same coherence decay; the fast channel code equals an explicit Kraus sum.
2. **Integrator accuracy.** The Strang error falls as n_s⁻²; the exact and Strang propagators
   agree with an independent adaptive Runge–Kutta solution of the master equation; n_s is set
   by the calibration of Section 2.2 (measured values below).
3. **Echo state property.** For every configuration, two Haar-random initial states driven by
   the same washout input must end within trace distance 10⁻⁶. Failures are counted, reported
   by channel and γ, and excluded. The washout length was set by a pilot at N = 4
   (`scripts/pilot_esp.py`): the share passing rises from 86% to 97% (amplitude damping) and
   from 57% to 74% (dephasing) between 500 and 2,000 steps. The distances are bimodal, so the
   result does not depend on the tolerance: 10⁻⁴ gives 100% and 78%.
4. **Reproduction of [14]** (`scripts/reproduce_palacios.py`): at N = 5 under bit-flip and
   phase-flip noise, the IPC–coherence relation is monotonic, negativity falls substantially
   before capacity drops, and the two channels collapse onto one curve.
5. **Baseline sanity.** The tuned ESN must beat the untuned ESN and the linear–quadratic
   reference.

**Measured values from the pilot** (details in [PILOT_RESULTS.md](PILOT_RESULTS.md)).

Strang steps per sub-interval (τ = 1) needed for trace distance < 10⁻⁴ after 100 periods:

| γ | 0.001 | 0.0027 | 0.0072 | 0.019 | 0.052 | 0.14 | 0.37 | 1 |
|---|---|---|---|---|---|---|---|---|
| Pure dephasing | 8 | 8 | 8 | 8 | 16 | 16 | 16 | 32 |
| Amplitude damping | 8 | 8 | 8 | 16 | 16 | 16 | 32 | 64 |

Reference gap ratio ⟨r⟩ at N = 10 for the chosen levels (100 realisations per point), with
the regime label it implies:

| | W = 0.1 | W = 5 | W = 20 |
|---|---|---|---|
| h = 0.1 | 0.406 (other: near-classical, not localised) | 0.513 (ergodic) | 0.408 (localised) |
| h = 1 | 0.531 (ergodic) | 0.502 (ergodic) | 0.405 (localised) |
| h = 10 | 0.519 (ergodic) | 0.444 (other) | 0.398 (localised) |


## 2.11 Reproducibility — `sweep.py`

Hypotheses, margins, the parameter grid and the analysis plan are pre-registered after the
pilot and before the main sweep. Every random draw derives from one base seed through
`numpy.random.SeedSequence` keyed by the grid indices, so any single configuration can be
recomputed bit for bit (tested). Each sweep writes a provenance file with its configuration,
configuration hash, git commit, package versions and the n_s table. Results are stored as
Parquet tables, and every figure and table is produced from them by `scripts/analyze.py`.
Exact package versions are pinned in `requirements-lock.txt`.

## 2.12 Changes from draft v1 and open decisions

See [DESIGN_V2.md](DESIGN_V2.md): Section 3 lists every change from draft v1 with its reason,
Section 4 what the redesign does not change, and Section 5 the steps before pre-registration.
Two corrections concern the description of [14] in the paper text: its main encoding is the
mixed state diag(s, 1 − s), not the pure state of [5], and its noise is a continuous Lindblad
term quoted as an error probability, not a discrete Pauli channel.
